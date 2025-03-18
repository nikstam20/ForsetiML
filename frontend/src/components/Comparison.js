import React, { useState, useMemo } from "react";
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
  Divider,
  IconButton,
  Popover,
  Paper,
} from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import TuneIcon from "@mui/icons-material/Tune";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Card,
  CardContent,
} from "@mui/material";

const weightMarks = [
  { value: 0, label: "Very Low" },
  { value: 25, label: "Low" },
  { value: 50, label: "Medium" },
  { value: 75, label: "High" },
  { value: 100, label: "Very High" },
];

const experimentColors = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
];

function extractMetrics(experiments) {
  return experiments.map((exp, index) => {
    const initialMetrics = exp.initialMetrics || {};
    const resultMetrics = exp.resultMetrics || {};

    const utilityMetrics = ["accuracy", "precision", "recall", "f1_score"].map(
      (metric) => ({
        name: metric.toUpperCase(),
        value: resultMetrics[metric] || initialMetrics[metric] || 0,
      })
    );

    const fairnessMetrics = Object.keys(
      initialMetrics.fairness_metrics || {}
    ).map((key) => ({
      name: key.replace(/_/g, " ").toUpperCase(),
      value: resultMetrics[key] || initialMetrics.fairness_metrics[key] || 0,
    }));

    return {
      index,
      utilityMetrics,
      fairnessMetrics,
      allMetrics: [...utilityMetrics, ...fairnessMetrics].reduce(
        (acc, metric) => {
          acc[metric.name] = metric.value;
          return acc;
        },
        {}
      ),
    };
  });
}

function computeRankings(experiments, selectedMetrics, metricWeights) {
  if (experiments.length === 0 || selectedMetrics.length === 0) return [];

  return experiments.sort((a, b) => {
    let scoreA = selectedMetrics.reduce(
      (sum, metric) =>
        sum + (a.allMetrics[metric] || 0) * (metricWeights[metric] || 1),
      0
    );
    let scoreB = selectedMetrics.reduce(
      (sum, metric) =>
        sum + (b.allMetrics[metric] || 0) * (metricWeights[metric] || 1),
      0
    );
    return scoreB - scoreA;
  });
}

function Comparison({ experiments = [] }) {
  const [selectedExperiments, setSelectedExperiments] = useState(
    experiments.map((_, i) => i)
  );
  const extractedExperiments = useMemo(
    () => extractMetrics(experiments),
    [experiments]
  );
  const filteredExperiments = extractedExperiments.filter((exp) =>
    selectedExperiments.includes(exp.index)
  );
  const [expanded, setExpanded] = useState(false);

  const allAvailableMetrics = useMemo(() => {
    if (filteredExperiments.length === 0) return [];
    return Object.keys(filteredExperiments[0].allMetrics);
  }, [filteredExperiments]);

  const [selectedMetrics, setSelectedMetrics] = useState(allAvailableMetrics);
  const [metricWeights, setMetricWeights] = useState(() =>
    allAvailableMetrics.reduce((acc, metric) => ({ ...acc, [metric]: 1 }), {})
  );

  const rankings = useMemo(
    () => computeRankings(filteredExperiments, selectedMetrics, metricWeights),
    [filteredExperiments, selectedMetrics, metricWeights]
  );

  const lineChartData = useMemo(() => {
    return allAvailableMetrics.map((metric) => {
      const dataPoint = { metric };
      filteredExperiments.forEach((exp) => {
        dataPoint[`Exp ${exp.index + 1}`] = exp.allMetrics[metric];
      });
      return dataPoint;
    });
  }, [filteredExperiments, allAvailableMetrics]);

  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState(null);
  const handleOpen = (event, metric) => {
    setAnchorEl(event.currentTarget);
    setSelectedMetric(metric);
  };
  const handleClose = () => setAnchorEl(null);
  const handleWeightChange = (value) => {
    setMetricWeights((prev) => ({ ...prev, [selectedMetric]: value }));
    handleClose();
  };

  return (
    <Paper
      sx={{ mt: 5, p: 3, borderRadius: 2, bgcolor: "white", boxShadow: 3 }}
    >
      <Typography
        variant="h5"
        sx={{ textAlign: "center", fontWeight: "bold", mb: 3 }}
      >
        Comparative Analysis
      </Typography>

      {/* Experiment Selection */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6">Select Experiments</Typography>
        <Typography
          variant="caption"
          sx={{ color: "gray", mb: 2, display: "block" }}
        >
          Toggle experiments to include them in the comparison.
        </Typography>
        {experiments.map((exp, index) => (
          <FormControlLabel
            key={index}
            control={
              <Checkbox
                checked={selectedExperiments.includes(index)}
                onChange={() => {
                  setSelectedExperiments((prev) =>
                    prev.includes(index)
                      ? prev.filter((i) => i !== index)
                      : [...prev, index]
                  );
                }}
              />
            }
            label={`Experiment ${index + 1}`}
          />
        ))}
      </Box>

      {/* Graph Section */}
      <Box sx={{ display: "flex", gap: 3 }}>
        <Box sx={{ flex: 3, p: 2 }}>
          <Typography variant="h6" sx={{ textAlign: "center", mb: 1 }}>
            Metric Performance
          </Typography>
          <LineChart width={800} height={450} data={lineChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" tick={{ fontSize: 0 }} tickLine={false} />
            <YAxis />
            <Tooltip />
            <Legend />
            {filteredExperiments.map((exp, i) => (
              <Line
                key={exp.index}
                type="monotone"
                dataKey={`Exp ${exp.index + 1}`}
                stroke={experimentColors[i % experimentColors.length]}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </Box>

        {/* Experiment Ranking & Weights */}
        <Box sx={{ flex: 1.2, p: 2, bgcolor: "#F5F5F5", borderRadius: 2 }}>
          <Typography variant="h6" sx={{ textAlign: "center", mb: 2 }}>
            Experiment Ranking
          </Typography>

          {rankings.map((rank, i) => (
            <Card
              key={rank.index}
              sx={{
                mb: 1,
                p: 1,
                textAlign: "center",
                borderLeft: `5px solid ${
                  experimentColors[i % experimentColors.length]
                }`,
              }}
            >
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: "bold" }}>
                  #{i + 1}
                </Typography>
                <Typography variant="body1">
                  Experiment {rank.index + 1}
                </Typography>
              </CardContent>
            </Card>
          ))}

          {/* Metric Weights  */}
          <Accordion
            sx={{ mt: 3 }}
            expanded={expanded}
            onChange={() => setExpanded(!expanded)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6" sx={{ textAlign: "center" }}>
                Adjust Metric Weights
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              {allAvailableMetrics.map((metric) => (
                <Box key={metric} sx={{ mb: 2, px: 2 }}>
                  {" "}
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    {metric}
                  </Typography>
                  <Slider
                    defaultValue={50}
                    step={null}
                    marks={weightMarks.map((mark) => ({
                      ...mark,
                      sx: {
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        minWidth: 0,
                      },
                    }))}
                    onChangeCommitted={(e, newValue) =>
                      handleWeightChange(metric, newValue)
                    }
                  />
                </Box>
              ))}
            </AccordionDetails>
          </Accordion>
        </Box>
      </Box>
    </Paper>
  );
}

export default Comparison;
