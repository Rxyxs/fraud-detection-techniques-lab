-- Vistas analiticas reales sobre outputs/fraud.sqlite -- no solo un volcado
-- de tablas: agregaciones y comparaciones entre modelos que responden
-- preguntas de negocio directamente en SQL.

-- 1) Ranking de modelos por reduccion de costo de negocio (la metrica que
--    realmente importa para decidir cual desplegar, no solo AUC).
DROP VIEW IF EXISTS v_model_ranking;
CREATE VIEW v_model_ranking AS
SELECT
    model,
    roc_auc,
    pr_auc,
    cost_at_best_threshold,
    cost_reduction_pct,
    RANK() OVER (ORDER BY cost_at_best_threshold ASC) AS rank_by_cost
FROM model_metrics
ORDER BY cost_at_best_threshold ASC;

-- 2) Desempeno por decil de score predicho (CatBoost): confirma que el
--    modelo ordena correctamente el riesgo, no solo que el AUC agregado es alto.
-- SQLite no permite reusar el alias de una funcion de ventana en el propio
-- GROUP BY del mismo SELECT (bug real encontrado al ejecutar, no al leer el
-- codigo) -- se resuelve con una subquery que materializa el decil primero.
DROP VIEW IF EXISTS v_catboost_decile_performance;
CREATE VIEW v_catboost_decile_performance AS
SELECT
    decile,
    COUNT(*) AS n_transactions,
    SUM(y_true) AS n_actual_fraud,
    ROUND(AVG(proba_catboost), 6) AS avg_predicted_proba,
    ROUND(1.0 * SUM(y_true) / COUNT(*), 4) AS actual_fraud_rate
FROM (
    SELECT y_true, proba_catboost, NTILE(10) OVER (ORDER BY proba_catboost) AS decile
    FROM predictions
)
GROUP BY decile
ORDER BY decile;

-- 3) Donde CatBoost y XGBoost DISCREPAN de forma notoria (diferencia de
--    probabilidad > 0.3): los casos limite que un analista humano deberia
--    revisar primero, no solo confiar en el ensamble ciegamente.
DROP VIEW IF EXISTS v_model_disagreement;
CREATE VIEW v_model_disagreement AS
SELECT
    y_true,
    proba_catboost,
    proba_xgboost,
    ABS(proba_catboost - proba_xgboost) AS abs_diff
FROM predictions
WHERE ABS(proba_catboost - proba_xgboost) > 0.3
ORDER BY abs_diff DESC;

-- 4) Resumen ejecutivo: una sola fila con las metricas de negocio clave.
DROP VIEW IF EXISTS v_executive_summary;
CREATE VIEW v_executive_summary AS
SELECT
    (SELECT model FROM v_model_ranking WHERE rank_by_cost = 1) AS best_model,
    (SELECT cost_at_best_threshold FROM v_model_ranking WHERE rank_by_cost = 1) AS best_model_cost,
    (SELECT COUNT(*) FROM v_model_disagreement) AS n_high_disagreement_cases,
    (SELECT COUNT(*) FROM predictions) AS n_total_test_transactions;
