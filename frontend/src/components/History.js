import React, { useState, useEffect } from "react";
import { useDataset } from "./DatasetContext";
import {
  Box,
  Typography,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  Stack,
  Divider,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DeleteIcon from "@mui/icons-material/Delete";
import InfoIcon from "@mui/icons-material/Info";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";
import { CircularProgress } from "@mui/material";

import Comparison from "./Comparison";

const PARAMETER_INFO = {
  max_depth:
    "Controls the maximum depth of trees (higher = more complex models).",
  eta: "Learning rate (lower = slower but better training).",
  subsample: "Fraction of samples used for training (default: 1.0).",
  gamma: "Minimum loss reduction required to make a split.",
};

const DEFAULT_MODEL_PARAMS = [
  { key: "max_depth", value: 6 },
  { key: "eta", value: 0.3 },
  { key: "subsample", value: 1.0 },
  { key: "gamma", value: 0 },
];

function History() {
  const { experiments } = useDataset();
  const [localExperiments, setLocalExperiments] = useState(experiments || []);

  useEffect(() => {
    setLocalExperiments(experiments || []);
  }, [experiments]);

  const handleDelete = (index) => {
    const confirmDelete = window.confirm(
      `Are you sure you want to delete Experiment ${index + 1}?`
    );
    if (confirmDelete) {
      setLocalExperiments((prev) => prev.filter((_, i) => i !== index));
    }
  };

  return (
    <Box sx={{ p: 4, width: 1400, ml: -35 }}>
      {/* <Typography
        variant="h4"
        sx={{
          mb: 4,
          fontWeight: "bold",
          textAlign: "center",
          color: "#34568B",
        }}
      >
        Experiment History
      </Typography> */}

      {localExperiments.length === 0 ? (
        <Box sx={{ textAlign: "center", mt: 5 }}>
          <Typography variant="body1" sx={{ color: "#888" }}>
            No experiments saved yet. Start by running some experiments!
          </Typography>
        </Box>
      ) : (
        <>
          <Box
            sx={{
              maxHeight: 1000,
              overflowY: "auto",
              p: 2,
              bgcolor: "#F5F7FA",
              borderRadius: 2,
              boxShadow: 2,
            }}
          >
            <Typography
              variant="h5"
              sx={{ textAlign: "center", fontWeight: "bold", mb: 3 }}
            >
              Completed Experiments
            </Typography>

            {localExperiments.map((exp, index) => {
              const modelParams =
                exp.modelParams && exp.modelParams.length > 0
                  ? exp.modelParams
                  : [];
              const initialMetrics = exp.initialMetrics || {};
              const resultMetrics = exp.resultMetrics || {};
              const fairnessMetrics = Object.keys(
                initialMetrics.fairness_metrics || {}
              ).map((key) => ({
                name: key.replace(/_/g, " ").toUpperCase(),
                Before: initialMetrics.fairness_metrics?.[key] || 0,
                After: resultMetrics[key] || 0,
              }));

              return (
                <Accordion key={index} sx={{ borderRadius: 2, boxShadow: 1 }}>
                  <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    sx={{
                      bgcolor: "#F7FAFC",
                      display: "flex",
                      alignItems: "center",
                    }}
                  >
                    <Typography variant="h6">Experiment {index + 1}</Typography>
                    <Typography
                      variant="body2"
                      sx={{ ml: "auto", color: "#718096" }}
                    >
                      {exp.timestamp || "No timestamp available"}
                    </Typography>
                  </AccordionSummary>

                  <AccordionDetails>
                    {/* Model Parameters */}
                    <Typography
                      variant="body2"
                      sx={{ fontWeight: "bold", mb: 1 }}
                    >
                      Model Parameters:
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(auto-fit, minmax(200px, 1fr))",
                        gap: 2,
                      }}
                    >
                      {modelParams.map((param, i) => (
                        <Paper
                          key={i}
                          sx={{
                            p: 2,
                            borderRadius: 2,
                            display: "flex",
                            alignItems: "center",
                            gap: 1,
                          }}
                        >
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: "bold" }}
                          >
                            {param.key}:
                          </Typography>
                          <Typography variant="body2">{param.value}</Typography>
                          <Tooltip title="More info about this parameter" arrow>
                            <InfoIcon
                              fontSize="small"
                              sx={{ color: "#34568B" }}
                            />
                          </Tooltip>
                        </Paper>
                      ))}
                    </Box>

                    {/* Mitigation Algorithm */}
                    {exp.mitigationAlgorithm && (
                      <Typography
                        variant="body2"
                        sx={{ mt: 2, fontWeight: "bold" }}
                      >
                        Mitigation Algorithm: {exp.mitigationAlgorithm}
                      </Typography>
                    )}

                    <Divider sx={{ my: 3 }} />

                    {/* Utility Metrics */}
                    <Typography
                      variant="h6"
                      sx={{ mb: 2, color: "#34568B", textAlign: "center" }}
                    >
                      Utility Metrics
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(auto-fit, minmax(200px, 1fr))",
                        gap: 2,
                      }}
                    >
                      {["accuracy", "precision", "recall", "f1_score"].map(
                        (metric) => {
                          const beforeValue = initialMetrics[metric] || 0;
                          const afterValue =
                            resultMetrics[metric] || beforeValue;
                          const change = afterValue - beforeValue;

                          return (
                            <Paper
                              key={metric}
                              sx={{
                                p: 3,
                                borderRadius: 2,
                                boxShadow: 2,
                                textAlign: "center",
                                backgroundColor: "#F7FAFC",
                              }}
                            >
                              <Typography
                                variant="body1"
                                sx={{ fontWeight: "bold", mb: 1 }}
                              >
                                {metric.toUpperCase()}
                              </Typography>
                              <Box
                                sx={{
                                  position: "relative",
                                  width: "100%",
                                  height: 10,
                                  backgroundColor: "#E0E0E0",
                                  borderRadius: "5px",
                                }}
                              >
                                <Box
                                  sx={{
                                    position: "absolute",
                                    width: `${beforeValue * 100}%`,
                                    height: "100%",
                                    backgroundColor: "#8884d8",
                                  }}
                                />
                                <Box
                                  sx={{
                                    position: "absolute",
                                    width: `${Math.abs(change) * 100}%`,
                                    height: "100%",
                                    left: `${beforeValue * 100}%`,
                                    backgroundColor:
                                      change > 0 ? "#82ca9d" : "#FF6F61",
                                  }}
                                />
                              </Box>
                              <Typography variant="body2" sx={{ mt: 1 }}>
                                Before: {beforeValue.toFixed(4)} | After:{" "}
                                {afterValue.toFixed(4)}
                              </Typography>
                            </Paper>
                          );
                        }
                      )}
                    </Box>

                    <Divider sx={{ my: 3 }} />

                    {/* Fairness Metrics */}
                    <Typography
                      variant="h6"
                      sx={{ mb: 2, color: "#34568B", textAlign: "center" }}
                    >
                      Bias Indexes
                    </Typography>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "center",
                        flexWrap: "wrap",
                        gap: 4,
                      }}
                    >
                      {fairnessMetrics.map((metric) => (
                        <Paper
                          key={metric.name}
                          sx={{
                            p: 3,
                            borderRadius: 3,
                            textAlign: "center",
                            boxShadow: 3,
                          }}
                        >
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: "bold", mb: 1 }}
                          >
                            {metric.name}
                          </Typography>

                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              gap: 3,
                            }}
                          >
                            {/* Before */}
                            <Box sx={{ textAlign: "center" }}>
                              <CircularProgress
                                variant="determinate"
                                value={metric.Before * 100}
                                size={70}
                                thickness={6}
                                sx={{ color: "#8884d8" }}
                              />
                              <Typography
                                variant="body2"
                                sx={{ mt: 1, fontWeight: "bold" }}
                              >
                                Before
                              </Typography>
                              <Typography variant="body2">
                                {(metric.Before * 100).toFixed(1)}%
                              </Typography>
                            </Box>

                            {/* After */}
                            <Box sx={{ textAlign: "center" }}>
                              <CircularProgress
                                variant="determinate"
                                value={metric.After * 100}
                                size={70}
                                thickness={6}
                                sx={{ color: "#82ca9d" }}
                              />
                              <Typography
                                variant="body2"
                                sx={{ mt: 1, fontWeight: "bold" }}
                              >
                                After
                              </Typography>
                              <Typography variant="body2">
                                {(metric.After * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          </Box>
                        </Paper>
                      ))}
                    </Box>

                    <Box sx={{ textAlign: "right" }}>
                      <IconButton
                        onClick={() => handleDelete(index)}
                        sx={{ color: "#FF6F61" }}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              );
            })}
          </Box>

          <Comparison experiments={localExperiments} />
        </>
      )}
    </Box>
  );
}

export default History;
