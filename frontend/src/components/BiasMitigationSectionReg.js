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
import ResultsSectionReg from "./ResultsSectionReg";
import { useDataset } from "./DatasetContext";

export default function BiasMitigationSectionReg({
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
        description:
          "Reduces disparate impact in data by modifying feature values.",
        action: "applyDisparateImpactRemover",
      },
      {
        name: "Feature Removal (Blinding)",
        description:
          "Removes sensitive features from the dataset to reduce bias during training.",
        action: "applyFeatureRemoval",
      },
      {
        name: "Reweighting",
        description:
          "Assigns weights to instances to ensure fair representation of sensitive groups.",
        action: "applyReweighting",
      },
    ],
    "In-processing": [
      {
        name: "Adversarial Debiasing",
        description:
          "Learns a fair model during training by adversarially minimizing bias.",
        action: "applyAdversarialDebiasing",
      },
      {
        name: "Prejudice Remover",
        description:
          "Implements a fairness-aware regularization technique to reduce bias.",
        action: "applyPrejudiceRemover",
      },
    ],
    "Post-processing": [
      {
        name: "Calibrated Equalized Odds Postprocessing",
        description:
          "Refines predicted scores to meet fairness criteria across groups.",
        action: "applyCalibratedEqualizedOddsPostprocessing",
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
        "http://127.0.0.1:5000/api/debias/apply_disparate_impact_remover_reg",
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
    applyFeatureRemoval: handleGenericMethod(
      "apply_feature_removal_reg",
      "Pre-processing"
    ),
    applyReweighting: handleGenericMethod(
      "apply_reweighting_reg",
      "Pre-processing"
    ),
    applyOptimizedPreprocessing: handleGenericMethod(
      "apply_optimized_preprocessing_reg",
      "Pre-processing"
    ),
    applyAdversarialDebiasing: handleGenericMethod(
      "apply_adversarial_debiasing_reg",
      "In-processing"
    ),
    applyFairGBM: handleGenericMethod("apply_fair_gbm_reg", "In-processing"),
    applyPrejudiceRemover: handleGenericMethod(
      "apply_prejudice_remover_reg",
      "In-processing"
    ),
    applyCalibratedEqualizedOddsPostprocessing: handleGenericMethod(
      "apply_calibrated_equalized_odds_postprocessing_reg",
      "Post-processing"
    ),
  };

  const renderMethods = (category) => {
    return methods[category].map((method, index) => (
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
