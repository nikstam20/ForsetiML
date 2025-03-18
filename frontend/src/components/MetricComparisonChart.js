import React from "react";
import { Box, Typography } from "@mui/material";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function MetricComparisonChart({ chartData }) {
  return (
    <Box
      sx={{
        mt: 4,
        p: 4,
        backgroundColor: "#fff",
        borderRadius: 3,
        boxShadow: 2,
      }}
    >
      <Typography
        variant="h6"
        sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
      >
        Metric Comparison: Before vs. After
      </Typography>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <ChartTooltip
            contentStyle={{
              backgroundColor: "#f5f5f5",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />
          <Legend wrapperStyle={{ fontSize: "14px" }} />
          <Bar
            dataKey="Before"
            fill="#8884d8"
            barSize={20}
            radius={[5, 5, 0, 0]}
          />
          <Bar
            dataKey="After"
            fill="#82ca9d"
            barSize={20}
            radius={[5, 5, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
}
