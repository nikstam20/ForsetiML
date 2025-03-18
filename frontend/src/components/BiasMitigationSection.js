import React, { useState } from "react";
import {
  Paper,
  Tabs,
  Tab,
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  IconButton,
  CircularProgress,
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import axios from "axios";
import MetricComparisonChart from "./MetricComparisonChart";
import ResultsSection from "./ResultsSection";
import { useDataset } from "./DatasetContext";

export default function BiasMitigationSection({
  metricsComputed,
  dataset,
  targetAttr,
  sensitiveAttr,
  privilegedGroup,
  unprivilegedGroup,
  setResultMetrics,
  modelParams,
  metrics,
}) {
  const [selectedTab, setSelectedTab] = useState(0);
  const [methodLoading, setMethodLoading] = useState(null);
  const {
    results,
    setResults,
    methodType,
    setMethodType,
    setExperiments,
    experiments,
    saveExperiment,
    queryResults,
  } = useDataset();

  const methods = {
    "Pre-processing": [
      {
        name: "Disparate Impact Remover",
        description: "Reduces disparate impact in data.",
        action: "applyDisparateImpactRemover",
      },
      {
        name: "Relabelling",
        description: "Modifies labels to balance data.",
        action: "applyRelabelData",
      },
      {
        name: "Feature Importance Suppression",
        description: "Suppresses important features to reduce bias.",
        action: "applyFeatureImportanceSuppression",
      },
      {
        name: "Correlation Suppression",
        description: "Removes features correlated with sensitive attributes.",
        action: "applyCorrelationSuppression",
      },
      {
        name: "Prevalence Sampling",
        description: "Samples groups to balance prevalence.",
        action: "applyPrevalenceSampling",
      },
      {
        name: "Optimized Preprocessing",
        description: "Applies optimized transformations for fairness.",
        action: "applyOptimizedPreprocessing",
      },
    ],
    "In-processing": [
      {
        name: "Adversarial Debiasing",
        description: "Learns a fair model during training.",
        action: "applyAdversarialDebiasing",
      },
      {
        name: "Prejudice Remover",
        description: "Implements a prejudice-removal regularizer.",
        action: "applyPrejudiceRemover",
      },
      {
        name: "Exponentiated Gradient Reduction",
        description:
          "Reduces bias using a series of gradient-based optimizations.",
        action: "applyExponentiatedGradientReduction",
      },
      {
        name: "GerryFair Classifier",
        description:
          "Ensures fairness by focusing on group fairness under a max-min fairness criterion.",
        action: "applyGerryFairClassifier",
      },
      {
        name: "Meta Fair Classification",
        description:
          "Leverages meta-learning to achieve fairness across different subgroups.",
        action: "applyMetaFairClassification",
      },
      {
        name: "Adversarially Robust Training",
        description:
          "Enhances model robustness against adversarial examples to promote fairness.",
        action: "applyAdversariallyRobustTraining",
      },
      {
        name: "Representation Neutralization",
        description:
          "Neutralizes effects of sensitive attributes in representations.",
        action: "applyRepresentationNeutralization",
      },
      {
        name: "FairVIC",
        description: "Implements a variational inference approach to fairness.",
        action: "applyFairVIC",
      },
    ],
    "Post-processing": [
      {
        name: "Equalized Odds Postprocessing",
        description:
          "Adjusts predictions to equalize positive rates across groups subject to equal true positive rates.",
        action: "applyEqualizedOddsPostprocessing",
      },
      {
        name: "Calibrated Equalized Odds Postprocessing",
        description:
          "Adjusts predictions to equalize positive rates across groups while preserving calibration.",
        action: "applyCalibratedEqualizedOddsPostprocessing",
      },
      {
        name: "Reject Option Classification",
        description:
          "Gives favorable outcomes to unprivileged groups and unfavorable to privileged groups within a confidence band around the decision boundary.",
        action: "applyRejectOptionClassification",
      },
    ],
  };

  const handleApplyDisparateImpactRemover = async () => {
    if (!dataset || !targetAttr || !sensitiveAttr) {
      alert(
        "Please provide a dataset, target attribute, and sensitive attribute."
      );
      return;
    }

    setMethodLoading("disparateImpactRemover");
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/debias/apply_disparate_impact_remover",
        {
          data_path: dataset,
          sensitive_attribute: sensitiveAttr,
          target_attribute: targetAttr,
          repair_level: 1.0,
          privileged_group: privilegedGroup,
          unprivilegedGroup: unprivilegedGroup,
          test_size: 0.2,
          random_state: 42,
        }
      );
      const { metrics: newMetrics } = response.data;
      console.log(response.data);
      setResults({ newMetrics });
      setResultMetrics(newMetrics);
      saveExperiment({
        modelParams,
        initialMetrics: metrics,
        mitigationAlgorithm: "Disparate Impact Remover",
        resultMetrics: newMetrics,
        timestamp: new Date()
          .toLocaleString("en-GB", {
            timeZone: "Europe/Athens",
            hour12: false,
          })
          .replace(",", ""),
      });
      alert(
        response.data.message ||
          "Disparate Impact Remover applied successfully!"
      );
    } catch (error) {
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setMethodLoading(null);
    }
  };

  const handleMethodClick = (method) => {
    const actionHandler = methodActions[method.action];
    if (actionHandler) {
      actionHandler();
    }
  };

  const handleGenericMethod = (method, category) => async () => {
    if (!dataset || !targetAttr || !sensitiveAttr) {
      alert(
        "Please provide a dataset, target attribute, and sensitive attribute."
      );
      return;
    }
    console.log(method);
    setMethodLoading(method.action);
    setMethodType(category);
    try {
      const response = await axios.post(
        `http://127.0.0.1:5000/api/debias/${method}`,
        {
          data_path: dataset,
          sensitive_attribute: sensitiveAttr,
          target_attribute: targetAttr,
          test_size: 0.2,
          random_state: 42,
          privileged_group: privilegedGroup,
          unprivileged_group: unprivilegedGroup,
        }
      );
      const { metrics: newMetrics } = response.data;
      console.log(response.data);
      setResults({ newMetrics });
      setResultMetrics(newMetrics);
      saveExperiment({
        modelParams,
        initialMetrics: metrics,
        mitigationAlgorithm: method.name,
        resultMetrics: newMetrics,
        timestamp: new Date()
          .toLocaleString("en-GB", {
            timeZone: "Europe/Athens",
            hour12: false,
          })
          .replace(",", ""),
      });
      alert(response.data.message || `${method} applied successfully!`);
    } catch (error) {
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setMethodLoading(null);
    }
  };
  const methodActions = {
    applyDisparateImpactRemover: handleApplyDisparateImpactRemover,
    applyRelabelData: handleGenericMethod(
      "apply_relabel_data",
      "Pre-processing"
    ),
    applyFeatureImportanceSuppression: handleGenericMethod(
      "apply_feature_importance_suppression",
      "Pre-processing"
    ),
    applyCorrelationSuppression: handleGenericMethod(
      "apply_correlation_suppression",
      "Pre-processing"
    ),
    applyPrevalenceSampling: handleGenericMethod(
      "apply_prevalence_sampling",
      "Pre-processing"
    ),
    applyLabelFlipping: handleGenericMethod(
      "apply_label_flipping",
      "Pre-processing"
    ),
    applyOptimizedPreprocessing: handleGenericMethod(
      "apply_optimized_preprocessing",
      "Pre-processing"
    ),
    applyAdversarialDebiasing: handleGenericMethod(
      "apply_adversarial_debiasing",
      "In-processing"
    ),
    applyPrejudiceRemover: handleGenericMethod(
      "apply_prejudice_remover",
      "In-processing"
    ),
    applyExponentiatedGradientReduction: handleGenericMethod(
      "apply_exponentiated_gradient_reduction",
      "In-processing"
    ),
    applyGerryFairClassifier: handleGenericMethod(
      "apply_gerry_fair_classifier",
      "In-processing"
    ),
    applyMetaFairClassification: handleGenericMethod(
      "apply_meta_fair_classification",
      "In-processing"
    ),
    applyAdversariallyRobustTraining: handleGenericMethod(
      "apply_adversarially_robust_training",
      "In-processing"
    ),
    applyRepresentationNeutralization: handleGenericMethod(
      "apply_representation_neutralization",
      "In-processing"
    ),
    applyFairVIC: handleGenericMethod("apply_fair_vic", "In-processing"),
    applyThresholdOptimizer: handleGenericMethod(
      "apply_threshold_optimizer",
      "Post-processing"
    ),
    applyEqualizedOddsPostprocessing: handleGenericMethod(
      "apply_equalized_odds_postprocessing",
      "Post-processing"
    ),
    applyCalibratedEqualizedOddsPostprocessing: handleGenericMethod(
      "apply_calibrated_equalized_odds_postprocessing",
      "Post-processing"
    ),
    applyDeterministicReranking: handleGenericMethod(
      "apply_deterministic_reranking",
      "Post-processing"
    ),
    applyRejectOptionClassification: handleGenericMethod(
      "apply_reject_option_classification",
      "Post-processing"
    ),
    applyGroupAwareThresholdAdaptation: handleGenericMethod(
      "apply_group_aware_threshold_adaptation",
      "Post-processing"
    ),
  };

  const renderMethods = (category) => {
    if (!queryResults || !queryResults.methods) return null;

    return methods[category]
      .filter((method) =>
        queryResults.methods.some((m) => m.name === method.name)
      )
      .map((method, index) => (
        <Card
          key={index}
          elevation={4}
          sx={{
            mb: 3,
            borderRadius: 3,
            transition: "transform 0.3s",
            "&:hover": { transform: "translateY(-5px)" },
          }}
        >
          <CardContent>
            <Typography
              variant="h6"
              sx={{ fontWeight: "bold", color: "#34568B" }}
            >
              {method.name}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, color: "#718096" }}>
              {method.description}
            </Typography>
          </CardContent>
          <CardActions>
            <IconButton
              color="primary"
              onClick={() => handleMethodClick(method)}
              disabled={methodLoading === method.action}
              sx={{
                border: "2px solid #34568B",
                borderRadius: "50%",
                backgroundColor:
                  methodLoading === method.action ? "#f0f0f0" : "#fff",
                "&:hover":
                  methodLoading === method.action
                    ? { backgroundColor: "#f0f0f0", cursor: "not-allowed" }
                    : { backgroundColor: "#34568B", color: "#fff" },
                transition: "all 0.3s ease-in-out",
              }}
            >
              {methodLoading === method.action ? (
                <CircularProgress size={24} sx={{ color: "#34568B" }} />
              ) : (
                <CheckCircleIcon />
              )}
            </IconButton>
          </CardActions>
        </Card>
      ));
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        mt: 4,
      }}
    >
      {metricsComputed ? (
        <>
          <Paper
            elevation={3}
            sx={{
              p: 4,
              border: "2px solid #f0d343",
              borderRadius: 3,
              backgroundColor: "#fff",
              mb: 4,
              width: "100%",
            }}
          >
            <Typography
              variant="h4"
              gutterBottom
              sx={{ fontWeight: "bold", color: "#34568B" }}
            >
              Bias Mitigation
            </Typography>

            <Tabs
              value={selectedTab}
              onChange={(e, newValue) => setSelectedTab(newValue)}
              textColor="primary"
              indicatorColor="primary"
              TabIndicatorProps={{
                style: {
                  height: 4,
                  backgroundColor: "#34568B",
                  transform: `translateX(${selectedTab * 32}%)`,
                  transition: "transform 0.3s ease",
                  width: "33.3333%",
                },
              }}
              sx={{
                mb: 3,
              }}
            >
              {Object.keys(methods).map((stage, index) => (
                <Tab
                  key={index}
                  label={stage}
                  sx={{
                    minWidth: 0,
                    textAlign: "center",
                    flex: 1,
                    padding: "12px 0",
                  }}
                />
              ))}
            </Tabs>

            <Box>
              {Object.keys(methods).map(
                (stage, index) => selectedTab === index && renderMethods(stage)
              )}
            </Box>
          </Paper>
        </>
      ) : (
        <Typography
          variant="h6"
          sx={{ fontWeight: "bold", color: "#34568B", textAlign: "center" }}
        />
      )}
    </Box>
  );
}
