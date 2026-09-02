// Reimplementacion en Rust puro (sin ONNX Runtime -- su binario prebuilt para
// Windows choco con el toolchain de link.exe disponible, simbolos
// vectorizados de C++20 sin resolver, un problema de compatibilidad MSVC
// documentado, no un error del modelo) de la inferencia del ensamble
// XGBoost afinado (726 arboles, exportado a JSON via `booster.save_model`).
// Mismo espiritu que la reimplementacion en C de chile-credit-risk-scoring-engine:
// reproducir bit-a-bit (dentro de tolerancia float) la prediccion de Python
// en un lenguaje de sistemas, y medir la latencia real de un hot-path de
// produccion.
//
//   cargo run --release

use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::time::Instant;

const N_FEATURES: usize = 37;

#[derive(Deserialize)]
struct XgbTree {
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<u32>,
    split_conditions: Vec<f32>,
}

struct Ensemble {
    trees: Vec<XgbTree>,
    base_score: f32,
}

impl Ensemble {
    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let raw = fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let learner = &v["learner"];
        let base_score: f32 = learner["learner_model_param"]["base_score"]
            .as_str()
            .unwrap()
            .parse()?;

        let trees_json = learner["gradient_booster"]["model"]["trees"].as_array().unwrap();
        let trees: Vec<XgbTree> = trees_json
            .iter()
            .map(|t| serde_json::from_value(t.clone()).unwrap())
            .collect();

        Ok(Ensemble { trees, base_score })
    }

    /// Traversal de un arbol: nodo hoja cuando left_children[nodo] == -1.
    /// Convencion XGBoost: si feature < split_condition, va a la izquierda.
    fn score_tree(tree: &XgbTree, row: &[f32; N_FEATURES]) -> f32 {
        let mut node = 0usize;
        loop {
            let left = tree.left_children[node];
            if left == -1 {
                return tree.split_conditions[node]; // valor de hoja
            }
            let feat_idx = tree.split_indices[node] as usize;
            let threshold = tree.split_conditions[node];
            node = if row[feat_idx] < threshold {
                left as usize
            } else {
                tree.right_children[node] as usize
            };
        }
    }

    /// base_score de XGBoost viene en espacio de probabilidad (~0.5); la
    /// suma de arboles esta en espacio de log-odds (margin). Se convierte
    /// base_score a logit, se suma el margin de los arboles, y se aplica
    /// sigmoid -- la formula exacta que usa XGBoost internamente para
    /// `binary:logistic`.
    fn predict_proba(&self, row: &[f32; N_FEATURES]) -> f32 {
        let logit_base = (self.base_score / (1.0 - self.base_score)).ln();
        let tree_sum: f32 = self.trees.iter().map(|t| Self::score_tree(t, row)).sum();
        sigmoid(logit_base + tree_sum)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("[1/4] Cargando ensamble XGBoost (726 arboles) desde JSON...");
    let ensemble = Ensemble::load("../../outputs/models/xgboost_tuned.json")?;
    println!("  {} arboles cargados, base_score={:.6}", ensemble.trees.len(), ensemble.base_score);

    println!("[2/4] Cargando transacciones reales (mismas filas de test que Python) y aplicando el mismo feature engineering...");
    let rows = load_rows("../../outputs/reports/rust_verification_rows.csv", 2000)?;
    println!("  {} filas reales cargadas", rows.len());

    println!("[3/4] Verificacion: comparando contra las probabilidades reales de Python (primeras 5 filas)...");
    let python_reference = load_python_reference("../../outputs/reports/rust_verification_reference.csv")?;
    let mut max_abs_diff = 0.0f32;
    for (i, (row, py_proba)) in rows.iter().zip(python_reference.iter()).enumerate().take(2000) {
        let rust_proba = ensemble.predict_proba(row);
        let diff = (rust_proba - py_proba).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        if i < 5 {
            println!("  fila {i}: python={py_proba:.6}  rust={rust_proba:.6}  diff={diff:.8}");
        }
    }
    println!("  Diferencia absoluta maxima sobre {} filas: {:.8}", rows.len(), max_abs_diff);

    println!("\n[4/4] Benchmark de latencia (1 fila a la vez, worst-case de scoring en tiempo real)...");
    let start = Instant::now();
    let mut fraud_count = 0usize;
    for row in &rows {
        if ensemble.predict_proba(row) >= 0.5 {
            fraud_count += 1;
        }
    }
    let elapsed = start.elapsed();
    let per_row_ns = elapsed.as_nanos() as f64 / rows.len() as f64;

    println!("\n=== Resultado ===");
    println!("Filas procesadas: {}", rows.len());
    println!("Predicciones de fraude: {}", fraud_count);
    println!("Tiempo total: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("Latencia por transaccion: {:.0}ns ({:.4}ms)", per_row_ns, per_row_ns / 1_000_000.0);
    println!("Throughput: {:.0} transacciones/segundo", 1_000_000_000.0 / per_row_ns);
    println!("Diferencia maxima vs. Python (726 arboles, 2000 filas reales): {:.8}", max_abs_diff);

    Ok(())
}

fn load_python_reference(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for result in reader.records() {
        let record = result?;
        out.push(record[0].parse::<f32>()?);
    }
    Ok(out)
}

fn load_rows(path: &str, limit: usize) -> Result<Vec<[f32; N_FEATURES]>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let idx = |name: &str| headers.iter().position(|h| h == name).unwrap();

    let v_idx: Vec<usize> = (1..=28).map(|i| idx(&format!("V{i}"))).collect();
    let amount_idx = idx("Amount");
    let top_cols = ["V14", "V12", "V4", "V11"];
    let top_idx: Vec<usize> = top_cols.iter().map(|c| idx(c)).collect();

    let mut rows = Vec::new();
    for (i, result) in reader.records().enumerate() {
        if i >= limit {
            break;
        }
        let record = result?;
        let v: Vec<f32> = v_idx.iter().map(|&j| record[j].parse::<f32>().unwrap()).collect();
        let amount: f32 = record[amount_idx].parse().unwrap();
        let amount_log = (1.0 + amount).ln();
        let v_l2_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let top_vals: Vec<f32> = top_idx.iter().map(|&j| record[j].parse::<f32>().unwrap()).collect();
        let mut interactions = Vec::new();
        for a in 0..top_vals.len() {
            for b in (a + 1)..top_vals.len() {
                interactions.push(top_vals[a] * top_vals[b]);
            }
        }

        let mut features = [0f32; N_FEATURES];
        features[0..28].copy_from_slice(&v);
        features[28] = amount;
        features[29] = amount_log;
        features[30] = v_l2_norm;
        features[31..37].copy_from_slice(&interactions);
        rows.push(features);
    }
    Ok(rows)
}
