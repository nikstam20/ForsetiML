import React from "react";
import { Typography, Paper, Box } from "@mui/material";
import ConfusionMatrix from "./ConfusionMatrix";
import UtilityMetrics from "./UtilityMetrics";
import FairnessMetrics from "./FairnessMetrics";
import MetricComparisonChart from "./MetricComparisonChart";

const InferenceMetrics = ({ metrics, resultMetrics, chartData }) => (
  <Paper elevation={3} sx={{ p: 4 }}>
    <Typography variant="h4" gutterBottom>
      Inference Metrics
    </Typography>
    <ConfusionMatrix matrix={metrics.confusion_matrix} />
    <UtilityMetrics metrics={metrics} />
    <FairnessMetrics metrics={metrics.fairness_metrics} />
    {resultMetrics && <MetricComparisonChart chartData={chartData} />}
  </Paper>
);

export default InferenceMetrics;
