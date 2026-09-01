# Modelo regulatorio: Regresion Logistica (GLM) sobre el mismo dataset real
# de fraude, como comparacion interpretable/auditable frente a los ensambles
# de arboles en Python -- el mismo patron que chile-credit-risk-scoring-engine
# (R para el modelo regulatorio, Python para los challengers ML).
#
# Sin dependencias externas (base R only): evita depender de instalar
# paquetes CRAN en una maquina sin ellos ya presentes.
#
#   Rscript r/logistic_model.R

set.seed(42)

cat("[1/5] Cargando dataset real...\n")
df <- read.csv("data/raw/creditcard_2023.csv")
cat(sprintf("  %d filas reales cargadas\n", nrow(df)))

pca_cols <- paste0("V", 1:28)
df$amount_log <- log1p(df$Amount)

cat("[2/5] Split train/test (80/20, estratificado)...\n")
idx_fraud <- which(df$Class == 1)
idx_legit <- which(df$Class == 0)
train_fraud <- sample(idx_fraud, floor(0.8 * length(idx_fraud)))
train_legit <- sample(idx_legit, floor(0.8 * length(idx_legit)))
train_idx <- c(train_fraud, train_legit)
test_idx <- setdiff(seq_len(nrow(df)), train_idx)

train_df <- df[train_idx, ]
test_df <- df[test_idx, ]
cat(sprintf("  train=%d  test=%d\n", nrow(train_df), nrow(test_df)))

cat("[3/5] Ajustando GLM (familia binomial, logit)...\n")
formula_str <- paste("Class ~", paste(c(pca_cols, "amount_log"), collapse = " + "))
model <- glm(as.formula(formula_str), data = train_df, family = binomial(link = "logit"))

cat("[4/5] Prediciendo sobre test held-out y calculando AUC (calculo manual, sin paquetes externos)...\n")
test_df$pred_proba <- predict(model, newdata = test_df, type = "response")

# AUC via regla del trapecio sobre la curva ROC (equivalente al estadistico
# de Mann-Whitney U / Wilcoxon rank-sum, calculado a mano deliberadamente
# para no depender de un paquete como pROC).
auc_manual <- function(labels, scores) {
  ord <- order(scores, decreasing = TRUE)
  labels <- labels[ord]
  n_pos <- sum(labels == 1)
  n_neg <- sum(labels == 0)
  tpr <- cumsum(labels == 1) / n_pos
  fpr <- cumsum(labels == 0) / n_neg
  tpr <- c(0, tpr)
  fpr <- c(0, fpr)
  sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
}

auc <- auc_manual(test_df$Class, test_df$pred_proba)
cat(sprintf("  AUC (test held-out) = %.4f\n", auc))

cat("[5/5] Guardando coeficientes, deviance y resultados...\n")
dir.create("outputs/reports", showWarnings = FALSE, recursive = TRUE)

coefs <- summary(model)$coefficients
coef_df <- data.frame(
  feature = rownames(coefs),
  estimate = coefs[, "Estimate"],
  std_error = coefs[, "Std. Error"],
  z_value = coefs[, "z value"],
  p_value = coefs[, "Pr(>|z|)"]
)
write.csv(coef_df, "outputs/reports/r_glm_coefficients.csv", row.names = FALSE)

n_significant <- sum(coef_df$p_value < 0.05)

# JSON escrito a mano (sin jsonlite): mantiene el script en base R puro,
# sin depender de que un paquete CRAN externo este instalado.
json_lines <- c(
  "{",
  sprintf('  "auc_test": %.4f,', auc),
  sprintf('  "null_deviance": %.2f,', model$null.deviance),
  sprintf('  "residual_deviance": %.2f,', model$deviance),
  sprintf('  "aic": %.2f,', model$aic),
  sprintf('  "n_train": %d,', nrow(train_df)),
  sprintf('  "n_test": %d,', nrow(test_df)),
  sprintf('  "n_significant_coefs_p005": %d', n_significant),
  "}"
)
writeLines(json_lines, "outputs/reports/r_glm_result.json")

cat("\n=== Resultado GLM (R) ===\n")
cat(sprintf("AUC test held-out: %.4f\n", auc))
cat(sprintf("Deviance nula -> residual: %.2f -> %.2f\n", model$null.deviance, model$deviance))
cat(sprintf("AIC: %.2f\n", model$aic))
cat(sprintf("Coeficientes significativos (p<0.05): %d / %d\n", n_significant, nrow(coef_df)))
cat("\nGuardado en: outputs/reports/r_glm_coefficients.csv, outputs/reports/r_glm_result.json\n")
