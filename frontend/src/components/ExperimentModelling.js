import React, { useState, useEffect } from "react";
import {
  Container,
  Typography,
  Paper,
  TextField,
  Button,
  CircularProgress,
  Grid2,
  IconButton,
  Tooltip,
  MenuItem,
  Slider,
  Box,
  Tab,
  Tabs,
} from "@mui/material";
import { Add, Remove, Info } from "@mui/icons-material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import axios from "axios";
import { useDataset } from "./DatasetContext";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
} from "recharts";
import DynamicForm from "./DynamicForm";
import FairnessMetricsChart from "./FairnessMetricsChart";
import { LinearProgress } from "@mui/material";
import BiasMitigationSection from "./BiasMitigationSection";
import ResultsSection from "./ResultsSection";
import CounterfactualAnalysis from "./CounterfactualAnalysis";

const theme = createTheme({
  palette: {
    primary: {
      main: "#34568B",
    },
    secondary: {
      main: "#6C8EBF",
    },
    background: {
      default: "#F7FAFC",
      paper: "#FFFFFF",
    },
    text: {
      primary: "#2D3748",
      secondary: "#718096",
    },
  },
  typography: {
    fontFamily: "'Roboto', Arial, sans-serif",
    fontWeightRegular: 400,
    fontWeightBold: 700,
  },
  shape: {
    borderRadius: 8,
  },
});

const parameterInfo = {
  booster: "Type of booster to use: 'gbtree', 'gblinear', or 'dart'.",
  eta: "Learning rate (default: 0.3). Lower values make learning slower but improve performance.",
  max_depth:
    "Maximum depth of the tree (default: 6). Deeper trees increase complexity.",
  min_child_weight:
    "Minimum sum of instance weight (hessian) in a child (default: 1).",
  gamma: "Minimum loss reduction required to split a node (default: 0).",
  subsample: "Fraction of samples used for training (default: 1.0).",
};

function ExperimentModelling() {
  const {
    dataset,
    sensitiveAttr,
    targetAttr,
    privilegedGroup,
    unprivilegedGroup,
    setUnprivilegedGroup,
    setSensitiveAttr,
    metrics,
    setMetrics,
    resultMetrics,
    setResultMetrics,
    chartData,
    setChartData,
    experiments,
    saveExperiment,
    attributes,
    setTargetAttr,
    results,
    baseTime,
    setBaseTime,
    baseSize,
    setBaseSize,
    methodType,
    setMethodType,
    userOptions,
    limitations,
  } = useDataset();

  const [loadingTrain, setLoadingTrain] = useState(false);
  const [loadingInference, setLoadingInference] = useState(false);
  const [modelParams, setModelParams] = useState([{ key: "", value: "" }]);
  const [testDataPath, setTestDataPath] = useState("");
  // const [metrics, setMetrics] = useState(null);
  // const [resultMetrics, setResultMetrics] = useState(null);
  const [splitPercentage, setSplitPercentage] = useState(80);
  const [loadingPreProcessing, setLoadingPreProcessing] = useState(false);
  const [loadingInProcessing, setLoadingInProcessing] = useState(false);
  const [loadingPostProcessing, setLoadingPostProcessing] = useState(false);
  const [lg, setLg] = useState(true);
  const [selectedTab, setSelectedTab] = useState(0);
  // const [chartData, setChartData] = useState([]);
  const [showInitialModal, setShowInitialModal] = useState(true);
  const [formSelections, setFormSelections] = useState({
    problemType: "",
    highLevelNotions: [],
    limitations: [],
  });
  const [activeTab, setActiveTab] = useState(0);
  const handleActiveTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  const [similarEntities, setSimilarEntities] = useState([]);
  const [counterfactuals, setCounterfactuals] = useState([]);
  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const handleSplitChange = (event, newValue) => {
    setSplitPercentage(newValue);
  };

  const handleTrainModel = async () => {
    if (!dataset || !targetAttr) {
      alert(
        "Please provide a dataset and select a target attribute for training."
      );
      return;
    }

    const params = modelParams.reduce((acc, { key, value }) => {
      if (key) acc[key] = value;
      return acc;
    }, {});

    setLoadingTrain(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/model/train_classification_model",
        {
          data_path: dataset,
          target_column: targetAttr,
          model_params: params,
          split_percentage: splitPercentage,
        }
      );
      setBaseTime(response.data.training_time_seconds);
      setBaseSize(response.data.model_size_bytes);
      alert(response.data.message || "Model trained successfully!");
    } catch (error) {
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoadingTrain(false);
    }
  };
  const fairnessMetricGroups = {
    independence: ["statistical_parity"],
    separation: [
      "equalized_odds",
      "equal_opportunity",
      "balance_for_positive_class",
      "balance_for_negative_class",
    ],
    sufficiency: [
      "predictive_equality",
      "conditional_use_accuracy_equality",
      "well_calibration",
      "test_fairness",
    ],
  };

  const handleInference = async () => {
    // if (!testDataPath) {
    //   alert("Please provide a valid test data path for inference.");
    //   return;
    // }

    setLoadingInference(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/model/make_classification_inference",
        {
          test_data_path: dataset,
          target_column: targetAttr,
          test_split_percent: 20,
          sensitive_feature: sensitiveAttr,
          limitations: limitations,
        }
      );
      const {
        predictions,
        probabilities,
        accuracy,
        precision,
        recall,
        f1_score,
        confusion_matrix: {
          true_negatives,
          false_positives,
          false_negatives,
          true_positives,
        },
        conditional_statistical_parity,
        conditional_use_accuracy_equality,
        equal_negative_predictive_value,
        equal_opportunity,
        overall_accuracy_equality,
        predictive_equality,
        statistical_parity,
        treatment_equality,
        balance_for_positive_class,
        equalized_odds,
        balance_for_negative_class,
        test_fairness,
        well_calibration,
      } = response.data;

      setMetrics({
        predictions,
        probabilities,
        accuracy,
        precision,
        recall,
        f1_score,
        confusion_matrix: {
          true_negatives,
          false_positives,
          false_negatives,
          true_positives,
        },
        fairness_metrics: filterFairnessMetrics(
          svlog({
            conditional_statistical_parity,
            conditional_use_accuracy_equality,
            equal_negative_predictive_value,
            equal_opportunity,
            overall_accuracy_equality,
            predictive_equality,
            statistical_parity,
            treatment_equality,
            balance_for_positive_class,
            equalized_odds,
            balance_for_negative_class,
            test_fairness,
            well_calibration,
          })
        ),
      });
    } catch (error) {
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      console.log(metrics);
      setLoadingInference(false);
    }
  };

  const handleAddParam = () =>
    setModelParams([...modelParams, { key: "", value: "" }]);

  const handleRemoveParam = (index) => {
    setModelParams(modelParams.filter((_, i) => i !== index));
  };

  const handleParamChange = (index, field, value) => {
    const updatedParams = [...modelParams];
    updatedParams[index][field] = value;
    setModelParams(updatedParams);
  };

  function filterFairnessMetrics(mtrs) {
    let selectedMetrics = {};
    console.log(userOptions);

    Object.keys(fairnessMetricGroups).forEach((category) => {
      console.log(userOptions);
      if (userOptions.includes(category)) {
        fairnessMetricGroups[category].forEach((metricKey) => {
          if (mtrs[metricKey] !== undefined) {
            selectedMetrics[metricKey] = mtrs[metricKey];
          }
        });
      }
    });

    return selectedMetrics;
  }
  const methods = {
    "Pre-processing": [
      {
        name: "Blinding: Feature Removal",
        description:
          "Removes sensitive features from the dataset to reduce bias during training.",
      },
      {
        name: "Synthetic Minority Oversampling Technique (SMOTE)",
        description:
          "Balances class distribution by oversampling minority groups to improve fairness.",
      },
      {
        name: "Reweighting",
        description:
          "Assigns weights to instances to ensure fair representation of sensitive groups.",
      },
      {
        name: "Disparate Impact Remover",
        description: "Reduces disparate impact by modifying feature values.",
      },
    ],
    "In-processing": [
      {
        name: "Adversarial Debiasing",
        description:
          "Simultaneously maximizes accuracy and minimizes bias in model predictions.",
      },
      {
        name: "FairGBM",
        description:
          "A boosting algorithm that incorporates fairness constraints during training.",
      },
      {
        name: "Prejudice Remover",
        description:
          "Adds a fairness regularization term to the model's loss function.",
      },
      {
        name: "FairVic",
        description: "Optimizes fairness directly within neural networks.",
      },
    ],
    "Post-processing": [
      {
        name: "Threshold Optimizer",
        description:
          "Adjusts prediction thresholds for different groups to improve fairness.",
      },
      {
        name: "Equalized Odds PostProcessing",
        description:
          "Modifies model outputs to satisfy equalized odds constraints.",
      },
      {
        name: "Calibrated Equalized Odds Postprocessing",
        description:
          "Refines predicted scores to meet fairness criteria across groups.",
      },
    ],
  };

  const metricKeys = [
    "accuracy",
    "conditional_statistical_parity",
    "conditional_use_accuracy_equality",
    "equal_negative_predictive_value",
    "equal_opportunity",
    "overall_accuracy_equality",
    "predictive_equality",
    "statistical_parity",
    "treatment_equality",
    "balance_for_positive_class",
    "equalized_odds",
    "balance_for_negative_class",
    "test_fairness",
    "well_calibration",
  ];

  useEffect(() => {
    if (metrics && resultMetrics) {
      const updatedChartData = metricKeys.map((key) => ({
        name: key.replace(/_/g, " ").toUpperCase(),
        Before: metrics[key] || 0,
        After: resultMetrics[key] || 0,
      }));
      setChartData(updatedChartData);
    }
  }, [metrics, resultMetrics, metricKeys, setChartData]);

  useEffect(() => {
    console.log("Form Selections:", formSelections);
  }, [formSelections]);
  const [counterfactualResults, setCounterfactualResults] = useState([]);
  const [counterfactualFairnessMetric, setCounterfactualFairnessMetric] =
    useState(null);

  function svlog(ff) {
    return Object.fromEntries(
      Object.entries(ff).map(([key, value]) => [key, value * 1])
    );
  }

  const [loadingCounterfactual, setLoadingCounterfactual] = useState(false);

  const handleFindSimilar = async (selectedEntityIndex) => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/counterfactual/find_similar",
        {
          data_path: dataset,
          target_column: targetAttr,
          selected_entity_index: selectedEntityIndex,
          num_neighbors: 5,
        }
      );
      setSimilarEntities(response.data.similar_entities);
    } catch (error) {
      alert(
        `Error finding similar entities: ${
          error.response?.data?.error || error.message
        }`
      );
    }
  };

  const handleGenerateCounterfactuals = async (selectedEntityIndex) => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/counterfactual/generate",
        {
          data_path: dataset,
          target_column: targetAttr,
          selected_entity_index: selectedEntityIndex,
          perturbation_fraction: 0.1,
          num_counterfactuals: 5,
        }
      );
      setCounterfactuals(response.data.counterfactuals);
    } catch (error) {
      alert(
        `Error generating counterfactuals: ${
          error.response?.data?.error || error.message
        }`
      );
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container
        maxWidth="lg"
        sx={{
          display: "flex",
          flexDirection: "column",
          flex: 1,
          transform: "scale(0.7)",
          transformOrigin: "top left",
          mt: -3,
          ml: -30,
        }}
      >
        <DynamicForm
          open={showInitialModal}
          onClose={() => setShowInitialModal(false)}
          onSubmit={(selections) => {
            setFormSelections(selections);
            setShowInitialModal(false);
          }}
        />

        {/* <Typography
            variant="h2"
            align="center"
            gutterBottom
            sx={{ fontWeight: "bold", color: theme.palette.primary.main }}
          >
            Experiment Modelling
          </Typography> */}

        {/* Main Grid Layout */}
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: "1fr 2fr",
            columnGap: 4,
            rowGap: 0,
            alignItems: "start",
            flex: 1,
          }}
        >
          {/* Column 1: Train and Test Model + Bias Mitigation Section */}
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              columnGap: 4,
              rowGap: -4,
              width: "540px",
            }}
          >
            {/* Train and Test Model */}
            <Paper
              elevation={3}
              sx={{
                p: 4,
                mb: 4,
                borderRadius: theme.shape.borderRadius,
                backgroundColor: theme.palette.background.paper,
                flex: 1,
              }}
            >
              <Typography variant="h4" gutterBottom>
                Train and Test Model
              </Typography>
              <Typography
                variant="body2"
                sx={{ mb: 2, color: theme.palette.text.secondary }}
              >
                Dataset: <strong>{dataset || "No dataset uploaded"}</strong>
              </Typography>
              {/* Target Column */}
              <TextField
                label="Target Column"
                variant="outlined"
                value={targetAttr}
                onChange={(e) => setTargetAttr(e.target.value)}
                select
                SelectProps={{ native: true }}
                sx={{ mb: 2, width: 300 }}
              >
                <option value=""></option>
                {attributes.map((attr, idx) => (
                  <option key={idx} value={attr}>
                    {attr}
                  </option>
                ))}
              </TextField>
              {/* Sensitive Attribute */}
              <TextField
                label="Sensitive Attribute"
                variant="outlined"
                value={sensitiveAttr}
                onChange={(e) => setSensitiveAttr(e.target.value)}
                select
                SelectProps={{ native: true }}
                sx={{ mb: 2, width: 300 }}
              >
                <option value=""></option>
                {attributes.map((attr, idx) => (
                  <option key={idx} value={attr}>
                    {attr}
                  </option>
                ))}
              </TextField>
              {/* Unprivileged Class */}
              <TextField
                label="Unprivileged Class"
                variant="outlined"
                value={unprivilegedGroup}
                onChange={(e) => setUnprivilegedGroup(e.target.value)}
                sx={{ mb: 2, width: 300 }}
              />
              {/* Train/Test Split */}
              <Typography variant="body1" sx={{ mb: 1 }}>
                Train/Test Split: {splitPercentage}% Training,{" "}
                {100 - splitPercentage}% Testing
              </Typography>
              <Slider
                value={splitPercentage}
                onChange={handleSplitChange}
                aria-labelledby="train-test-split-slider"
                valueLabelDisplay="auto"
                valueLabelFormatter={(value) => `${value}%`}
                min={50}
                max={90}
                step={5}
                sx={{ width: 300 }}
              />
              {/* Model Parameters */}
              <Typography variant="body1" sx={{ mb: 2 }}>
                Model Parameters:
              </Typography>
              {modelParams.map((param, index) => (
                <Grid2 container spacing={2} key={index} sx={{ mb: 2 }}>
                  <Grid2 item xs={4}>
                    <TextField
                      fullWidth
                      select
                      label="Key"
                      variant="outlined"
                      value={param.key}
                      sx={{ minWidth: 80 }}
                      onChange={(e) =>
                        handleParamChange(index, "key", e.target.value)
                      }
                    >
                      <MenuItem value="">Select Parameter</MenuItem>
                      {Object.keys(parameterInfo).map((paramKey) => (
                        <MenuItem key={paramKey} value={paramKey}>
                          {paramKey}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid2>

                  <Grid2 item xs={4}>
                    <TextField
                      fullWidth
                      label="Value"
                      variant="outlined"
                      value={param.value}
                      sx={{ width: "120px" }}
                      onChange={(e) =>
                        handleParamChange(index, "value", e.target.value)
                      }
                    />
                  </Grid2>

                  <Grid2 item xs={2}>
                    <Tooltip
                      title={
                        parameterInfo[param.key] || "No information available"
                      }
                    >
                      <IconButton sx={{ color: theme.palette.primary.main }}>
                        <Info />
                      </IconButton>
                    </Tooltip>
                  </Grid2>

                  <Grid2 item xs={2}>
                    {modelParams.length > 1 && (
                      <IconButton
                        sx={{ color: theme.palette.secondary.main }}
                        onClick={() => handleRemoveParam(index)}
                      >
                        <Remove />
                      </IconButton>
                    )}
                  </Grid2>
                </Grid2>
              ))}
              <Button
                variant="outlined"
                color="secondary"
                onClick={handleAddParam}
                sx={{ mb: 3, borderRadius: "12px", py: 1.5, width: 200 }}
              >
                <Add /> Add Parameter
              </Button>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  columnGap: 2,
                  mt: 3,
                }}
              >
                {/* Train Model Button */}
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleTrainModel}
                  disabled={loadingTrain}
                  sx={{
                    flex: 1,
                    borderRadius: "50px",
                    py: 1.5,
                    fontSize: "1rem",
                    fontWeight: 600,
                    textTransform: "none",
                    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
                    "&:hover": {
                      backgroundColor: "#2D4F82",
                      boxShadow: "0 6px 15px rgba(0, 0, 0, 0.2)",
                    },
                    transition: "all 0.3s ease-in-out",
                  }}
                >
                  {loadingTrain ? (
                    <CircularProgress size={24} sx={{ color: "#fff" }} />
                  ) : (
                    "Train Model"
                  )}
                </Button>

                {/* Make Inference Button */}
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleInference}
                  disabled={loadingInference}
                  sx={{
                    flex: 1,
                    borderRadius: "50px",
                    py: 1.5,
                    fontSize: "1rem",
                    fontWeight: 600,
                    textTransform: "none",
                    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
                    "&:hover": {
                      backgroundColor: "#2D4F82",
                      boxShadow: "0 6px 15px rgba(0, 0, 0, 0.2)",
                    },
                    transition: "all 0.3s ease-in-out",
                  }}
                >
                  {loadingInference ? (
                    <CircularProgress size={24} sx={{ color: "#fff" }} />
                  ) : (
                    "Make Inference"
                  )}
                </Button>
              </Box>
            </Paper>

            {/* Bias Mitigation Section */}
            <BiasMitigationSection
              metricsComputed={!!metrics}
              dataset={dataset}
              targetAttr={targetAttr}
              sensitiveAttr={sensitiveAttr}
              privilegedGroup={privilegedGroup}
              unprivilegedGroup={unprivilegedGroup}
              setResultMetrics={setResultMetrics}
              modelParams={modelParams}
              metrics={metrics}
              elevation={3}
              sx={{
                p: 4,
                border: "2px solid #f0d343",
                borderRadius: theme.shape.borderRadius,
                backgroundColor: theme.palette.background.paper,
                flex: 1,
              }}
            />
          </Box>

          {/* Column 2: Inference Results */}
          <Box sx={{ height: "100%" }}>
            {metrics && lg && (
              <Paper
                elevation={3}
                sx={{
                  p: 4,
                  borderRadius: theme.shape.borderRadius,
                  backgroundColor: theme.palette.background.paper,
                  flex: 1,
                }}
              >
                {/* Tabs for toggling between Inference Results and Individual Fairness Analysis */}
                <Tabs
                  value={activeTab}
                  onChange={(e, newValue) => setActiveTab(newValue)}
                  textColor="primary"
                  TabIndicatorProps={{ style: { display: "none" } }}
                  sx={{ mb: 3 }}
                >
                  <Tab label="Inference Results" />
                  <Tab label="Individual Fairness Analysis" />
                </Tabs>

                {/* Tab Panel 1: Inference Results */}
                {activeTab === 0 && metrics && (
                  <Box>
                    <Typography variant="h4" gutterBottom>
                      Inference Results
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1.5fr",
                        gap: 4,
                        height: "100%",
                      }}
                    >
                      <Box>
                        <ConfusionMatrix matrix={metrics.confusion_matrix} />
                        <UtilityMetrics metrics={metrics} />
                      </Box>
                      <Box sx={{ width: "600px" }}>
                        <FairnessMetrics metrics={metrics} />
                      </Box>
                    </Box>
                  </Box>
                )}

                {/* Tab Panel 2: Individual Fairness Analysis */}
                {activeTab === 1 && dataset && targetAttr && (
                  <Box sx={{ width: "1000px" }}>
                    <Typography variant="h4" gutterBottom>
                      Individual Fairness Analysis
                    </Typography>
                    <CounterfactualAnalysis
                      dataset={dataset}
                      targetAttr={targetAttr}
                      sensitiveAttr={sensitiveAttr}
                    />
                  </Box>
                )}
              </Paper>
            )}

            <Box sx={{ mt: 4, flex: 1 }}>
              {results && (
                <ResultsSection
                  newMetrics={results.newMetrics}
                  methodType={methodType}
                />
              )}
            </Box>
          </Box>
        </Box>
      </Container>
    </ThemeProvider>
  );
}
function UtilityMetrics({ metrics }) {
  // console.log("Utility Metrics:", metrics);
  const utilityMetrics = [
    { label: "Accuracy", value: metrics.accuracy, max: 1 },
    { label: "Precision", value: metrics.precision, max: 1 },
    { label: "Recall", value: metrics.recall, max: 1 },
    { label: "F1 Score", value: metrics.f1_score, max: 1 },
  ];

  return (
    <Box>
      <Typography
        variant="h6"
        sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
      >
        Utility Metrics
      </Typography>
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: 2,
        }}
      >
        {utilityMetrics.map((metric) => (
          <Paper
            key={metric.label}
            sx={{
              p: 2,
              borderRadius: 2,
              boxShadow: 2,
              textAlign: "center",
              backgroundColor: "#F7FAFC",
            }}
          >
            <Typography variant="body1" sx={{ fontWeight: "bold", mb: 1 }}>
              {metric.label}
            </Typography>
            <Tooltip title={`Value: ${(metric.value || 0).toFixed(4)}`}>
              <LinearProgress
                variant="determinate"
                value={(metric.value || 0) * 100}
                sx={{
                  height: 10,
                  borderRadius: 5,
                  backgroundColor: "#E0E0E0",
                  "& .MuiLinearProgress-bar": { backgroundColor: "#6C8EBF" },
                }}
              />
            </Tooltip>
            <Typography variant="body2" sx={{ mt: 1 }}>
              {(metric.value || 0).toFixed(4)} / {metric.max}
            </Typography>
          </Paper>
        ))}
      </Box>
    </Box>
  );
}

function ConfusionMatrix({ matrix }) {
  return (
    <Box sx={{ mb: 4 }}>
      <Typography
        variant="h6"
        sx={{
          fontWeight: "bold",
          mb: 2,
          color: "#34568B",
          textAlign: "left",
        }}
      >
        Confusion Matrix
      </Typography>
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          textAlign: "center",
          backgroundColor: "#F7FAFC",
          borderRadius: 3,
          boxShadow: 3,
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#E2E8F0",
            color: "#2D3748",
            p: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          Actual / Predicted
        </Box>
        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#6C8EBF",
            color: "white",
            p: 2,
          }}
        >
          Negative
        </Box>
        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#6C8EBF",
            color: "white",
            p: 2,
          }}
        >
          Positive
        </Box>

        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#34568B",
            color: "white",
            p: 2,
          }}
        >
          Negative
        </Box>
        <Box
          sx={{
            backgroundColor: "#C8E6C9",
            fontWeight: "bold",
            color: "#2D3748",
            p: 2,
          }}
        >
          {matrix.true_negatives}
        </Box>
        <Box
          sx={{
            backgroundColor: "#FFCDD2",
            fontWeight: "bold",
            color: "#2D3748",
            p: 2,
          }}
        >
          {matrix.false_positives}
        </Box>

        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#34568B",
            color: "white",
            p: 2,
          }}
        >
          Positive
        </Box>
        <Box
          sx={{
            backgroundColor: "#FFCDD2",
            fontWeight: "bold",
            color: "#2D3748",
            p: 2,
          }}
        >
          {matrix.false_negatives}
        </Box>
        <Box
          sx={{
            backgroundColor: "#C8E6C9",
            fontWeight: "bold",
            color: "#2D3748",
            p: 2,
          }}
        >
          {matrix.true_positives}
        </Box>
      </Box>
    </Box>
  );
}

function FairnessMetrics({ metrics }) {
  // console.log(metrics);
  return (
    <Box>
      <Typography
        variant="h6"
        sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
      >
        Bias Indexes
      </Typography>
      <FairnessMetricsChart metrics={metrics} />
    </Box>
  );
}

function MetricComparisonChart({ chartData }) {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography
        variant="h6"
        sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
      >
        Metric Comparison (Before vs. After)
      </Typography>
      <BarChart
        width={800}
        height={400}
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <ChartTooltip />
        <Legend />
        <Bar dataKey="Before" fill="#8884d8" />
        <Bar dataKey="After" fill="#82ca9d" />
      </BarChart>
    </Box>
  );
}

export default ExperimentModelling;
