import React from "react";
import { Typography, Paper, Box, LinearProgress, Tooltip } from "@mui/material";

const UtilityMetrics = ({ metrics }) => {
  const utilityMetrics = [
    { label: "Accuracy", value: metrics.accuracy, max: 1 },
    { label: "Precision", value: metrics.precision, max: 1 },
    { label: "Recall", value: metrics.recall, max: 1 },
    { label: "F1 Score", value: metrics.f1_score, max: 1 },
  ];

  return (
    <Box>
      <Typography variant="h6" sx={{ fontWeight: "bold", mb: 2 }}>
        Utility Metrics
      </Typography>
      <Box sx={{ display: "grid", gap: 2 }}>
        {utilityMetrics.map((metric) => (
          <Paper
            key={metric.label}
            sx={{ p: 2, textAlign: "center", boxShadow: 1 }}
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
            <Typography variant="body2">
              {(metric.value || 0).toFixed(4)} / {metric.max}
            </Typography>
          </Paper>
        ))}
      </Box>
    </Box>
  );
};

export default UtilityMetrics;
