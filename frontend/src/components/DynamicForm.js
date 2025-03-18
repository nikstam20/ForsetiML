import React, { useState } from "react";
import {
  Modal,
  Box,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  LinearProgress,
  Tooltip,
  Tabs,
  Tab,
  createTheme,
  ThemeProvider,
  Grid2,
  Container,
} from "@mui/material";
import {
  Gavel as FairnessIcon,
  Equalizer as MetricIcon,
  Assignment as LimitationIcon,
  CheckCircle as ReviewIcon,
  Info as InfoIcon,
} from "@mui/icons-material";
import "./DynamicForm.css";
import { useDataset } from "./DatasetContext";

const theme = createTheme({
  palette: {
    primary: { main: "#3c8dbc" },
    secondary: { main: "#f0d343" },
    background: { paper: "#f9f9f9", default: "#ffffff" },
  },
  typography: {
    fontFamily: "Inter, Roboto, Arial, sans-serif",
    fontWeightBold: 600,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: "8px",
          textTransform: "none",
          fontWeight: 500,
        },
      },
    },
    MuiStepIcon: {
      styleOverrides: {
        root: {
          "&.Mui-completed": { color: "#f0d343" },
          "&.Mui-active": { color: "#3c8dbc" },
        },
      },
    },
  },
});

const steps = [
  { label: "Problem Type", icon: <FairnessIcon /> },
  { label: "Fairness Goals", icon: <MetricIcon /> },
  { label: "Specify Limitations", icon: <LimitationIcon /> },
  { label: "Final Details", icon: <ReviewIcon /> },
];

const fairnessNotions = [
  {
    label: "Independence",
    value: "independence",
    description: "Ensure predictions are independent of protected attributes.",
  },
  {
    label: "Separation",
    value: "separation",
    description:
      "Maintain equal false positive and false negative rates across groups.",
  },
  {
    label: "Sufficiency",
    value: "sufficiency",
    description:
      "Predictions should be sufficient without sensitive attributes.",
  },
  {
    label: "Loss",
    value: "loss",
    description: "Minimizes error rate disparities across groups.",
  },
];

const limitations = [
  {
    id: 33,
    keyword: "Possible Loss of Content",
    description:
      "Some information might be left out during the process, impacting the overall result.",
    category: "Data-Related",
  },
  {
    id: 34,
    keyword: "Indirect Bias",
    description:
      "Bias could be introduced indirectly through hidden patterns in the data.",
    category: "Data-Related",
  },
  {
    id: 35,
    keyword: "Proxy Variables",
    description:
      "Variables that stand in for sensitive attributes might cause unintended fairness issues.",
    category: "Data-Related",
  },
  {
    id: 36,
    keyword: "Complexity",
    description:
      "The approach may be hard to understand or implement for all users.",
    category: "Model-Related",
  },
  {
    id: 37,
    keyword: "Model Performance",
    description:
      "Achieving fairness might affect the accuracy or efficiency of the model.",
    category: "Model-Related",
  },
  {
    id: 38,
    keyword: "Overfitting",
    description:
      "The model might fit the training data too closely, making it less effective for new data.",
    category: "Model-Related",
  },
  {
    id: 40,
    keyword: "Intersectional Fairness",
    description:
      "The method might not account for fairness across multiple sensitive attributes.",
    category: "Fairness",
  },
  {
    id: 42,
    keyword: "Assumption of Linearity",
    description:
      "Fairness methods might assume a linear relationship, which could be inaccurate for some data.",
    category: "Fairness",
  },
  {
    id: 43,
    keyword: "Reliance on Data",
    description:
      "The fairness method relies heavily on the quality and quantity of the data provided.",
    category: "Performance",
  },
  {
    id: 44,
    keyword: "Reliance on Fairness Constraints",
    description:
      "Too much dependency on fairness constraints might limit flexibility.",
    category: "Fairness",
  },
  {
    id: 45,
    keyword: "Conflicting Fairness Requirements",
    description: "Meeting one fairness goal might conflict with another.",
    category: "Fairness",
  },
  {
    id: 46,
    keyword: "Reliance on Parameters",
    description:
      "The method requires careful tuning of parameters for effectiveness.",
    category: "Performance",
  },
  {
    id: 47,
    keyword: "Computational Cost",
    description: "The method could require significant time or resources.",
    category: "Performance",
  },
  {
    id: 48,
    keyword: "Reliance on Estimator",
    description:
      "Certain methods rely on specific estimators, limiting generalizability.",
    category: "Performance",
  },
  {
    id: 49,
    keyword: "Suitability",
    description: "Not all fairness methods are suitable for every use case.",
    category: "Performance",
  },
];

const conflicts = [
  {
    id: 1,
    keyword: "Fairness-Accuracy Trade-off",
    description:
      "Improving fairness can lead to decreased accuracy, creating a trade-off between performance and ethics.",
  },
  {
    id: 2,
    keyword: "Transparency Issues",
    description:
      "Certain fairness methods may reduce the transparency of the model or data, making its decisions harder to understand and raising security concerns.",
  },
  {
    id: 3,
    keyword: "Impact on Interpretability",
    description:
      "Techniques such as complex neural networks can improve fairness but make model predictions harder to explain.",
  },
  {
    id: 4,
    keyword: "Conflicts Between Fairness Metrics",
    description:
      "Different fairness metrics assess fairness in different ways, sometimes making it impossible to satisfy all of them simultaneously.",
  },
];

function getStepContent(
  stepIndex,
  input,
  handleChange,
  currentTab,
  handleTabChange
) {
  const availableNotions = fairnessNotions.filter((notion) => {
    if (input.problemType === "classification") return notion.value !== "loss";
    if (input.problemType === "regression")
      return notion.value !== "sufficiency";
    return true;
  });

  switch (stepIndex) {
    case 0:
      return (
        <Box>
          <Typography
            variant="h5"
            sx={{ fontWeight: "bold", textAlign: "center", mb: 2 }}
          >
            What type of problem are you wishing to solve?
          </Typography>
          <Box display="flex" justifyContent="center" gap={6}>
            {[
              {
                type: "classification",
                label: "Classification",
                icon: (
                  <img
                    src="/classification-icon.png"
                    alt="Classification"
                    style={{ width: "130px", height: "130px" }}
                  />
                ),
              },
              {
                type: "regression",
                label: "Regression",
                icon: (
                  <img
                    src="/regression-analysis.png"
                    alt="Regression"
                    style={{ width: "130px", height: "130px" }}
                  />
                ),
              },
            ].map(({ type, label, icon }) => (
              <Box
                key={type}
                onClick={() => handleChange("problemType", type)}
                sx={{
                  cursor: "pointer",
                  borderRadius: 3,
                  width: "240px",
                  height: "260px",
                  textAlign: "center",
                  p: 4,
                  border:
                    input.problemType === type
                      ? "4px solid #3c8dbc"
                      : "3px solid #e0e0e0",
                  boxShadow: input.problemType === type ? 6 : 3,
                  backgroundColor:
                    input.problemType === type ? "#e8f4fd" : "#f9f9f9",
                  transition: "transform 0.3s, box-shadow 0.3s",
                  "&:hover": {
                    transform: "translateY(-8px)",
                    boxShadow: 8,
                  },
                }}
              >
                {icon}
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: "bold",
                    mt: 3,
                    color: input.problemType === type ? "#3c8dbc" : "#000",
                  }}
                >
                  {label}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      );
    case 1:
      return (
        <Box>
          <Typography
            variant="h5"
            sx={{ fontWeight: "bold", textAlign: "center", mb: 2 }}
          >
            Which of these notions comply with your objectives? Pick up to
            three.
          </Typography>
          <Box
            display="flex"
            flexWrap="wrap"
            justifyContent="center"
            gap={2}
            sx={{ maxWidth: "500px", mx: "auto" }}
          >
            {availableNotions.map((notion) => (
              <Card
                key={notion.value}
                onClick={() => handleChange("highLevelNotions", notion.value)}
                sx={{
                  cursor: "pointer",
                  borderRadius: 2,
                  width: "200px",
                  textAlign: "center",
                  border: input.highLevelNotions.includes(notion.value)
                    ? "3px solid #3c8dbc"
                    : "2px solid #e0e0e0",
                  boxShadow: input.highLevelNotions.includes(notion.value)
                    ? 4
                    : 2,
                  "&:hover": { boxShadow: 6 },
                }}
              >
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: "bold", mb: 1 }}>
                    {notion.label}
                  </Typography>
                  <Typography variant="body2">{notion.description}</Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        </Box>
      );
    case 2:
      return (
        <Box>
          <Typography
            variant="h5"
            sx={{ fontWeight: "bold", textAlign: "center", mb: 2 }}
          >
            Specify Limitations or Concerns
          </Typography>

          {/* Tabs for Limitations Categories */}
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            centered
            textColor="primary"
            sx={{
              mb: 3,
              "& .MuiTabs-flexContainer": { justifyContent: "center" },
              "& .MuiTab-root": {
                fontWeight: "bold",
                textTransform: "capitalize",
                px: 2,
                "&.Mui-selected": {
                  backgroundColor: "#e8f4fd",
                  borderRadius: 2,
                },
              },
            }}
          >
            {["Data-Related", "Model-Related", "Fairness", "Performance"].map(
              (category) => (
                <Tab key={category} label={category} value={category} />
              )
            )}
          </Tabs>

          {/* Limitations Selection */}
          <Box
            display="grid"
            gridTemplateColumns="repeat(auto-fit, minmax(250px, 1fr))"
            gap={2}
          >
            {limitations
              .filter((limitation) => limitation.category === currentTab)
              .map((limitation) => (
                <Box
                  key={limitation.id}
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 1,
                    p: 3,
                    borderRadius: 2,
                    textAlign: "center",
                    boxShadow: input.limitations.includes(limitation.keyword)
                      ? 4
                      : 2,
                    border: input.limitations.includes(limitation.keyword)
                      ? "3px solid #3c8dbc"
                      : "2px solid #e0e0e0",
                    backgroundColor: input.limitations.includes(
                      limitation.keyword
                    )
                      ? "#e8f4fd"
                      : "#f9f9f9",
                    cursor: "pointer",
                    transition: "transform 0.3s, box-shadow 0.3s",
                    "&:hover": {
                      transform: "translateY(-5px)",
                      boxShadow: 6,
                    },
                  }}
                  onClick={() =>
                    handleChange("limitations", limitation.keyword)
                  }
                >
                  <Typography
                    variant="body1"
                    sx={{
                      fontWeight: "bold",
                      mb: 1,
                      color: input.limitations.includes(limitation.keyword)
                        ? "#3c8dbc"
                        : "#000",
                    }}
                  >
                    {limitation.keyword}
                  </Typography>
                  <Tooltip title={limitation.description} arrow>
                    <InfoIcon fontSize="small" sx={{ color: "#3c8dbc" }} />
                  </Tooltip>
                </Box>
              ))}
          </Box>

          <Typography
            variant="h5"
            sx={{ fontWeight: "bold", textAlign: "center", mt: 4, mb: 2 }}
          >
            Potential Conflicts
          </Typography>

          <Box
            display="grid"
            gridTemplateColumns="repeat(auto-fit, minmax(250px, 1fr))"
            gap={2}
          >
            {conflicts.map((conflict) => (
              <Box
                key={conflict.id}
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 1,
                  p: 3,
                  borderRadius: 2,
                  textAlign: "center",
                  boxShadow: input.conflicts.includes(conflict.keyword) ? 4 : 2,
                  border: input.conflicts.includes(conflict.keyword)
                    ? "3px solid #d9534f"
                    : "2px solid #e0e0e0",
                  backgroundColor: input.conflicts.includes(conflict.keyword)
                    ? "#f9d6d5"
                    : "#f9f9f9",
                  cursor: "pointer",
                  transition: "transform 0.3s, box-shadow 0.3s",
                  "&:hover": {
                    transform: "translateY(-5px)",
                    boxShadow: 6,
                  },
                }}
                onClick={() => handleChange("conflicts", conflict.keyword)}
              >
                <Typography
                  variant="body1"
                  sx={{
                    fontWeight: "bold",
                    mb: 1,
                    color: input.conflicts.includes(conflict.keyword)
                      ? "#d9534f"
                      : "#000",
                  }}
                >
                  {conflict.keyword}
                </Typography>
                <Tooltip title={conflict.description} arrow>
                  <InfoIcon fontSize="small" sx={{ color: "#d9534f" }} />
                </Tooltip>
              </Box>
            ))}
          </Box>
        </Box>
      );

    case 3:
      return (
        <Box>
          <Typography
            variant="h5"
            sx={{ fontWeight: "bold", textAlign: "center", mb: 2 }}
          >
            Review and confirm your selections
          </Typography>
          <Typography sx={{ textAlign: "center" }}>
            This is the final step before submitting your experiment
            configuration.
          </Typography>
        </Box>
      );
    default:
      return "Unknown step";
  }
}

function DynamicForm({ onSubmit, onClose }) {
  const [open, setOpen] = useState(true);
  const [activeStep, setActiveStep] = useState(0);
  const [input, setInput] = useState({
    problemType: "",
    highLevelNotions: [],
    limitations: [],
    conflicts: [],
  });
  const [currentTab, setCurrentTab] = useState("Data-Related");

  const { setQueryResults, setUserOptions, setLimitations, setProbType } =
    useDataset();

  const handleNext = () => setActiveStep((prev) => prev + 1);
  const handleBack = () => setActiveStep((prev) => prev - 1);
  const handleChange = (name, value) => {
    setInput((prev) => {
      const newValue = Array.isArray(prev[name])
        ? prev[name].includes(value)
          ? prev[name].filter((item) => item !== value)
          : [...prev[name], value]
        : value;
      return { ...prev, [name]: newValue };
    });
  };
  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const handleSubmit = async () => {
    const requestData = {
      limitations: input.limitations,
      conflicts: [],
      fairness_notions: input.highLevelNotions,
    };
    setLimitations(input.limitations);
    setProbType(input.problemType);
    try {
      const response = await fetch(
        "http://127.0.0.1:5000/api/knowledge-graph/query",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestData),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch results");
      }

      const data = await response.json();
      setQueryResults(data);
    } catch (error) {
      console.error("Error fetching results:", error);
      alert("Error fetching results. Please try again.");
    }
    setUserOptions(input.highLevelNotions);
    setOpen(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <Modal open={open} onClose={onClose || (() => setOpen(false))}>
        <Box
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 1100,
            height: 700,
            bgcolor: "background.paper",
            borderRadius: 2,
            boxShadow: 6,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {/* Progress Bar */}
          <LinearProgress
            variant="determinate"
            value={(activeStep / steps.length) * 100}
            sx={{
              height: 8,
              backgroundColor: "#E0E0E0",
              "& .MuiLinearProgress-bar": {
                backgroundColor: "#3c8dbc",
              },
            }}
          />

          {/* Modal Content */}
          <Box sx={{ p: 4, flex: 1, overflowY: "auto" }}>
            <Typography
              variant="h4"
              gutterBottom
              sx={{
                textAlign: "center",
                fontWeight: "bold",
                color: theme.palette.primary.main,
              }}
            >
              Experiment Setup
            </Typography>
            <Stepper activeStep={activeStep} alternativeLabel>
              {steps.map((step, index) => (
                <Step key={index} completed={activeStep > index}>
                  <StepLabel>{step.label}</StepLabel>
                </Step>
              ))}
            </Stepper>

            {/* Content for Current Step */}
            <Box sx={{ mt: 3 }}>
              {getStepContent(
                activeStep,
                input,
                handleChange,
                currentTab,
                handleTabChange
              )}
            </Box>
          </Box>

          {/* Modal Footer */}
          <Box
            sx={{
              p: 2,
              display: "flex",
              justifyContent: "space-between",
              bgcolor: theme.palette.background.default,
              borderTop: "1px solid #E0E0E0",
            }}
          >
            <Button
              onClick={handleBack}
              disabled={activeStep === 0}
              sx={{
                color: theme.palette.primary.main,
                textTransform: "none",
                fontWeight: "bold",
              }}
            >
              Back
            </Button>
            <Button
              variant="contained"
              onClick={
                activeStep === steps.length - 1 ? handleSubmit : handleNext
              }
              sx={{
                background: "linear-gradient(45deg, #6faed9, #508bb5)", // Subtle gradient
                color: "#fff",
                borderRadius: "8px",
                fontWeight: "bold",
                textTransform: "none",
                "&:hover": {
                  background: "linear-gradient(45deg, #508bb5, #3c6c91)", // Slightly darker on hover
                },
              }}
            >
              {activeStep === steps.length - 1 ? "Finish" : "Next"}
            </Button>
          </Box>
        </Box>
      </Modal>
    </ThemeProvider>
  );
}

export default DynamicForm;
