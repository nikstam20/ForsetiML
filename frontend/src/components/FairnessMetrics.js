import React from "react";
import { Box, Typography } from "@mui/material";
import FairnessMetricsChart from "./FairnessMetricsChart";

const FairnessMetrics = ({ metrics }) => {
  return (
    <Box>
      <Typography variant="h6" sx={{ fontWeight: "bold", mb: 2 }}>
        Fairness Metrics
      </Typography>
      <Box sx={{ p: 2, backgroundColor: "#F7FAFC", borderRadius: 2 }}>
        <FairnessMetricsChart metrics={metrics} />
      </Box>
    </Box>
  );
};

export default FairnessMetrics;
