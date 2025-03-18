import React from "react";
import {
  Box,
  Typography,
  Paper,
  Tooltip,
  LinearProgress,
  Grid,
} from "@mui/material";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
} from "recharts";
import { useDataset } from "./DatasetContext";
import { alignProperty } from "@mui/material/styles/cssUtils";
import ArrowDropUpIcon from "@mui/icons-material/ArrowDropUp";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";

const COLORS = ["#FF6F61", "#E0E0E0"]; // Colors for donut chart

export default function ResultsSection({ newMetrics, methodType }) {
  const { metrics, baseTime, baseSize } = useDataset();
  if (!metrics || !newMetrics) {
    return (
      <Typography
        variant="body1"
        sx={{ color: "#ff0000", mt: 2, textAlign: "center" }}
      >
        Metrics data is unavailable.
      </Typography>
    );
  }

  const utilityMetricsKeys = ["accuracy", "precision", "recall", "f1_score"];
  const fairnessMetricsKeys = Object.keys(metrics.fairness_metrics || {});
  const modelSizeChange = ((newMetrics.model_size - baseSize) / baseSize) * 100;
  const trainingTimeChange =
    ((newMetrics.training_time - baseTime) / baseTime) * 100;
  const formatChange = (change) => ({
    value: Math.abs(change.toFixed(2)),
    icon:
      change >= 0 ? (
        <ArrowDropUpIcon style={{ color: "#4CAF50" }} />
      ) : (
        <ArrowDropDownIcon style={{ color: "#FF5722" }} />
      ),
  });

  const modelSize = formatChange(modelSizeChange);
  const trainingTime = formatChange(trainingTimeChange);

  const fairnessChartData = fairnessMetricsKeys.map((key) => ({
    name: key.replace(/_/g, " ").toUpperCase(),
    Before: metrics.fairness_metrics[key] || 0,
    After: newMetrics[key] || 0,
  }));

  const alterationRatio = newMetrics.alteration_ratio || 0;
  const dataAlterationChartData = [
    { name: "Altered", value: alterationRatio },
    { name: "Unaltered", value: 1 - alterationRatio },
  ];

  return (
    <Box
      sx={{
        textAlign: "center",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          borderRadius: 3,
          border: "2px solid #f0d343",
          backgroundColor: "#F7FAFC",
        }}
      >
        <Grid container spacing={4}>
          {/* Row 1: Fairness Metrics Comparison */}
          <Grid item xs={12}>
            <Typography
              variant="h6"
              sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
            >
              Fairness Indexes Comparison
            </Typography>
            <BarChart
              layout="vertical"
              width={900}
              height={800}
              data={fairnessChartData}
              margin={{ top: 20, right: 30, left: 40, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 10, fill: "#555", fontWeight: "bold" }}
              />
              <ChartTooltip />
              <Legend />
              <Bar dataKey="Before" fill="#8884d8" />
              <Bar
                dataKey="After"
                fill="#82ca9d"
                style={{ border: "6px solid #f0d343" }}
              />
            </BarChart>
          </Grid>

          {/* Row 2: Utility Metrics  */}
          <Grid container spacing={2}>
            {/* Left Column: Utility Metrics */}
            <Grid item xs={12} md={6}>
              <Box sx={{ textAlign: "center", pl: 2 }}>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
                >
                  Utility Metrics
                </Typography>
                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                    gap: 2,
                  }}
                >
                  {utilityMetricsKeys.map((key) => {
                    const beforeValue = metrics[key] || 0;
                    const afterValue = newMetrics[key] || 0;
                    const change = afterValue - beforeValue;
                    const isIncrease = change > 0;

                    return (
                      <Paper
                        key={key}
                        sx={{
                          p: 3,
                          pl: 4,
                          borderRadius: 2,
                          boxShadow: 2,
                          textAlign: "center",
                          backgroundColor: "#F7FAFC",
                        }}
                      >
                        <Typography
                          variant="body1"
                          sx={{ fontWeight: "bold", mb: 1, color: "#34568B" }}
                        >
                          {key.replace(/_/g, " ").toUpperCase()}
                        </Typography>

                        <Tooltip title={`Before: ${beforeValue.toFixed(4)}`}>
                          <LinearProgress
                            variant="determinate"
                            value={beforeValue * 100}
                            sx={{
                              height: 10,
                              borderRadius: 5,
                              backgroundColor: "#E0E0E0",
                              "& .MuiLinearProgress-bar": {
                                backgroundColor: "#8884d8",
                              },
                            }}
                          />
                        </Tooltip>

                        <Typography
                          variant="body2"
                          sx={{ mt: 1, color: "#718096" }}
                        >
                          Before: {beforeValue.toFixed(4)}
                        </Typography>

                        <Tooltip title={`After: ${afterValue.toFixed(4)}`}>
                          <LinearProgress
                            variant="determinate"
                            value={afterValue * 100}
                            sx={{
                              mt: 1,
                              height: 10,
                              borderRadius: 5,
                              backgroundColor: "#E0E0E0",
                              "& .MuiLinearProgress-bar": {
                                backgroundColor: "#82ca9d",
                              },
                            }}
                          />
                        </Tooltip>

                        <Typography
                          variant="body2"
                          sx={{ mt: 1, color: "#718096" }}
                        >
                          After: {afterValue.toFixed(4)}
                        </Typography>

                        <Box
                          sx={{
                            mt: 2,
                            p: 1,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: 1,
                            borderRadius: 2,
                            backgroundColor: isIncrease ? "#E6F9E5" : "#FFE5E5",
                          }}
                        >
                          <Box
                            component="span"
                            sx={{
                              fontSize: "1.5rem",
                              color: isIncrease ? "#82ca9d" : "#FF6F61",
                            }}
                          >
                            {isIncrease ? "↑" : "↓"}
                          </Box>
                          <Typography
                            variant="body2"
                            sx={{
                              fontWeight: "bold",
                              fontSize: "1rem",
                              color: isIncrease ? "#82ca9d" : "#FF6F61",
                            }}
                          >
                            {`${(change * 100).toFixed(2)}%`}
                          </Typography>
                        </Box>
                      </Paper>
                    );
                  })}
                </Box>
              </Box>
            </Grid>

            {/* Right Column: Model Properties or Data Distortion */}
            <Grid item xs={12} md={6}>
              <Box sx={{ textAlign: "center", width: "100%" }}>
                {methodType === "In-processing" ? (
                  <Box>
                    <Typography
                      variant="h6"
                      sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
                    >
                      Model Properties
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(auto-fit, minmax(250px, 1fr))",
                        gap: 3,
                        justifyContent: "center",
                        alignItems: "center",
                      }}
                    >
                      <Paper
                        elevation={3}
                        sx={{
                          p: 3,
                          borderRadius: 3,
                          textAlign: "center",
                          background:
                            modelSize.value > 100
                              ? "linear-gradient(to right, #FFE5E5, #FFB3B3)"
                              : "linear-gradient(to right, #E6F9E5, #B3F4B3)",
                          boxShadow:
                            modelSize.value > 100
                              ? "0 4px 10px rgba(255, 111, 97, 0.3)"
                              : "0 4px 10px rgba(130, 202, 157, 0.3)",
                        }}
                      >
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            mb: 2,
                          }}
                        >
                          <Box
                            component="span"
                            sx={{
                              fontSize: "2rem",
                              color:
                                modelSize.value > 100 ? "#FF6F61" : "#82ca9d",
                            }}
                          >
                            {modelSize.value > 100 ? "↑" : "↓"}
                          </Box>
                          <Typography
                            variant="h5"
                            sx={{
                              ml: 1,
                              fontWeight: "bold",
                              color:
                                modelSize.value > 100 ? "#FF6F61" : "#82ca9d",
                            }}
                          >
                            {`${modelSize.value}%`}
                          </Typography>
                        </Box>
                        <Typography
                          variant="body2"
                          sx={{ fontWeight: "bold", color: "#718096" }}
                        >
                          Model Size Compared to Baseline
                        </Typography>
                      </Paper>

                      <Paper
                        elevation={3}
                        sx={{
                          p: 3,
                          borderRadius: 3,
                          textAlign: "center",
                          background:
                            trainingTime.value > 100
                              ? "linear-gradient(to right, #FFE5E5, #FFB3B3)"
                              : "linear-gradient(to right, #E6F9E5, #B3F4B3)",
                          boxShadow:
                            trainingTime.value > 100
                              ? "0 4px 10px rgba(255, 111, 97, 0.3)"
                              : "0 4px 10px rgba(130, 202, 157, 0.3)",
                        }}
                      >
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            mb: 2,
                          }}
                        >
                          <Box
                            component="span"
                            sx={{
                              fontSize: "2rem",
                              color:
                                trainingTime.value > 100
                                  ? "#FF6F61"
                                  : "#82ca9d",
                            }}
                          >
                            {trainingTime.value > 100 ? "↑" : "↓"}
                          </Box>
                          <Typography
                            variant="h5"
                            sx={{
                              ml: 1,
                              fontWeight: "bold",
                              color:
                                trainingTime.value > 100
                                  ? "#FF6F61"
                                  : "#82ca9d",
                            }}
                          >
                            {`${trainingTime.value}%`}
                          </Typography>
                        </Box>
                        <Typography
                          variant="body2"
                          sx={{ fontWeight: "bold", color: "#718096" }}
                        >
                          Training Time Compared to Baseline
                        </Typography>
                      </Paper>
                    </Box>
                  </Box>
                ) : (
                  // Data Distortion Section
                  <Box sx={{ mt: 4, textAlign: "center", width: "100%" }}>
                    <Typography
                      variant="h6"
                      sx={{ fontWeight: "bold", mb: 2, color: "#34568B" }}
                    >
                      Data Distortion
                    </Typography>
                    <PieChart
                      width={200}
                      height={200}
                      style={{ margin: "auto" }}
                    >
                      <Pie
                        data={dataAlterationChartData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        fill="#82ca9d"
                        startAngle={90}
                        endAngle={-270}
                      >
                        {dataAlterationChartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={COLORS[index % COLORS.length]}
                          />
                        ))}
                      </Pie>
                      <text
                        x="50%"
                        y="50%"
                        textAnchor="middle"
                        dominantBaseline="middle"
                        style={{
                          fontSize: "24px",
                          fontWeight: "bold",
                          fill: "#FF6F61",
                        }}
                      >
                        {`${(alterationRatio * 100).toFixed(1)}%`}
                      </text>
                    </PieChart>
                  </Box>
                )}
              </Box>
            </Grid>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
}
