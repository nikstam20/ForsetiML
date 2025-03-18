import React, { useState } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  LineChart,
  Line,
} from "recharts";
import {
  Paper,
  Typography,
  Box,
  Tabs,
  Tab,
  Button,
  ButtonGroup,
} from "@mui/material";

const FairnessMetricsChart = ({ metrics }) => {
  const [chartType, setChartType] = useState("radar");

  const fairnessData = Object.entries(metrics.fairness_metrics || {}).map(
    ([key, value]) => {
      const { best, worst } = getMetricDetails(key);
      const normalizedValue = ((value - worst) / (best - worst)) * 100;
      return {
        metric: formatMetricName(key),
        value: normalizedValue,
        rawValue: value,
        best,
        worst,
      };
    }
  );

  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        borderRadius: 4,
        background: "linear-gradient(135deg, #F7FAFC 0%, #E2E8F0 100%)",
        boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.15)",
      }}
    >
      {/* <Typography
        variant="h6"
        sx={{ fontWeight: "bold", mb: 3, textAlign: "center", color: "#34568B" }}
      >
        Fairness Metrics Visualization
      </Typography> */}

      {/* Button Group for Chart Type Selection */}
      <Box sx={{ display: "flex", justifyContent: "center", mb: 3 }}>
        <ButtonGroup color="primary" variant="outlined">
          <Button
            onClick={() => setChartType("radar")}
            variant={chartType === "radar" ? "contained" : "outlined"}
          >
            Radar Chart
          </Button>
          <Button
            onClick={() => setChartType("bar")}
            variant={chartType === "bar" ? "contained" : "outlined"}
          >
            Bar Chart
          </Button>
          <Button
            onClick={() => setChartType("line")}
            variant={chartType === "line" ? "contained" : "outlined"}
          >
            Line Chart
          </Button>
        </ButtonGroup>
      </Box>

      {/* Chart Display */}
      <Box sx={{ height: 400 }}>
        <ResponsiveContainer width="100%" height="100%">
          {chartType === "radar" && (
            <RadarChart data={fairnessData} outerRadius="80%">
              <PolarGrid stroke="#CBD5E0" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{
                  fontSize: 10,
                  fill: "#2D3748",
                  fontWeight: "bold",
                  textAnchor: "middle",
                }}
                tickFormatter={(name) => splitMetricName(name)}
              />
              <PolarRadiusAxis
                angle={30}
                domain={[0, 100]}
                tick={{ fontSize: 10, fill: "#718096" }}
                axisLine={{ stroke: "#CBD5E0" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Radar
                name="Fairness Metric"
                dataKey="value"
                stroke="#6C8EBF"
                fill="#6C8EBF"
                fillOpacity={0.7}
              />
            </RadarChart>
          )}

          {chartType === "bar" && (
            <BarChart data={fairnessData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="metric"
                tickFormatter={(name) => splitMetricName(name)}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar dataKey="value" fill="#6C8EBF" />
            </BarChart>
          )}

          {chartType === "line" && (
            <LineChart data={fairnessData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="metric"
                tickFormatter={(name) => splitMetricName(name)}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#6C8EBF"
                strokeWidth={2}
                dot={{ fill: "#6C8EBF", r: 5 }}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
};

const CustomTooltip = ({ payload }) => {
  if (payload && payload.length > 0) {
    const data = payload[0].payload;
    return (
      <Paper
        sx={{
          padding: "8px",
          borderRadius: "8px",
          backgroundColor: "#2D3748",
          color: "#F7FAFC",
          boxShadow: "0px 2px 8px rgba(0,0,0,0.2)",
        }}
      >
        <Typography variant="body2" sx={{ fontWeight: "bold" }}>
          {data.metric}
        </Typography>
        <Typography variant="body2">
          Raw Value: {data.rawValue.toFixed(4)}
        </Typography>
        <Typography variant="body2">
          Normalized: {data.value.toFixed(2)} / 100
        </Typography>
      </Paper>
    );
  }
  return null;
};

const getMetricDetails = (metricName) => {
  return { best: 1, worst: 0 };
};

const formatMetricName = (name) =>
  name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

const splitMetricName = (name) => {
  const words = name.split(" ");
  return words.length > 2
    ? [words.slice(0, 2).join(" "), words.slice(2).join(" ")]
    : name;
};

export default FairnessMetricsChart;
