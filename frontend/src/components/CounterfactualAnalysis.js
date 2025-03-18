import React, { useState } from "react";
import {
  Box,
  Typography,
  Paper,
  Button,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
} from "@mui/material";
import axios from "axios";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import ActionableRecourse from "./ActionableRecourse";

const clusterColors = [
  "#FF6B6B",
  "#4D96FF",
  "#59CE8F",
  "#FF85A1",
  "#A680FF",
  "#FFD56B",
  "#4BC6B9",
  "#FFA07A",
];

export default function CounterfactualAnalysis({
  dataset,
  targetAttr,
  sensitiveAttr,
}) {
  const [loadingClusters, setLoadingClusters] = useState(false);
  const [clusters, setClusters] = useState({});
  const [possibleSensitiveValues, setPossibleSensitiveValues] = useState([]);
  const [selectedEntity, setSelectedEntity] = useState("");
  const [newSensitiveValue, setNewSensitiveValue] = useState("");
  const [loadingCounterfactual, setLoadingCounterfactual] = useState(false);
  const [counterfactual, setCounterfactual] = useState(null);
  const [loadingRecourse, setLoadingRecourse] = useState(false);
  const [recourse, setRecourse] = useState(null);
  const [viewMode, setViewMode] = useState("counterfactual");
  const [activeCluster, setActiveCluster] = useState(null);
  const [entityIdInput, setEntityIdInput] = useState("");

  const handleClusterEntities = async () => {
    setLoadingClusters(true);
    setClusters({});
    setPossibleSensitiveValues([]);
    setCounterfactual(null);
    setRecourse(null);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/counterfactual/cluster_entities",
        {
          data_path: dataset,
          target_column: targetAttr,
          sensitive_attribute: sensitiveAttr,
          num_clusters: 5,
        }
      );

      const data = response.data;
      if (!data.clusters) {
        console.error("No 'clusters' key found in response.data");
        alert("No clusters found in the response.");
        return;
      }

      setClusters(data.clusters);
      setPossibleSensitiveValues(data.possible_sensitive_values);
    } catch (error) {
      console.error("Error during API call:", error);
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoadingClusters(false);
    }
  };

  const handleFetchEntityById = async () => {
    if (!entityIdInput) {
      alert("Please enter a valid entity ID.");
      return;
    }

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/counterfactual/fetch_entity",
        {
          data_path: dataset,
          target_column: targetAttr,
          entity_id: parseInt(entityIdInput, 10),
        }
      );

      setSelectedEntity(response.data.entity);
      console.log(response.data.entity);
    } catch (error) {
      alert(`Error: ${error.response?.data?.error || error.message}`);
    }
  };

  const handleGenerateCounterfactual = async () => {
    // if (!newSensitiveValue || !selectedEntity) {
    //   alert("Please select an entity and a new sensitive value.");
    //   return;
    // }
    setLoadingCounterfactual(true);
    setCounterfactual(null);
    setRecourse(null);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/counterfactual/generate",
        {
          data_path: dataset,
          target_column: targetAttr,
          selected_entity_index: selectedEntity.index,
          sensitive_attribute: sensitiveAttr,
          new_sensitive_value: newSensitiveValue,
        }
      );
      setCounterfactual(response.data.counterfactual);
    } catch (error) {
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoadingCounterfactual(false);
      setNewSensitiveValue("");
    }
  };

  const handleLegendClick = (clusterName) => {
    const clusterKey = clusterName.split(" ")[1];
    setActiveCluster(activeCluster === clusterKey ? null : clusterKey);
  };

  const renderTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const entity = payload[0].payload;
      return (
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="body2">
            <strong>Entity #{entity.index}</strong>
          </Typography>
          <Typography variant="body2">
            <strong>Prediction:</strong> {entity.prediction}
          </Typography>
          <Typography variant="body2">
            <strong>Features:</strong>
            <pre style={{ margin: 0 }}>
              {JSON.stringify(entity.features, null, 2)}
            </pre>
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  const renderScatterPlots = () => {
    return Object.keys(clusters).map((clusterKey, index) => {
      const clusterEntities = clusters[clusterKey];

      return (
        <Scatter
          key={clusterKey}
          name={`Cluster ${clusterKey}`}
          data={clusterEntities}
          fill={
            activeCluster === null || activeCluster === clusterKey
              ? clusterColors[index % clusterColors.length]
              : "#ddd"
          }
          onClick={(e) => setSelectedEntity(e.payload)}
        />
      );
    });
  };

  return (
    <Box sx={{ p: 4 }}>
      {/* Clustering and Entity Selection Section */}
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" sx={{ mb: 2 }}>
          Clustering and Entity Selection
        </Typography>

        <Typography variant="body2" sx={{ color: "grey.600", mb: 2 }}>
          Clusters are generated using t-SNE for dimensionality reduction and
          KMeans for grouping. You can either pinpoint similar entities or input
          your own entity ID to fetch a specific entity.
        </Typography>

        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
          <Button
            variant="outlined"
            color="primary"
            onClick={handleClusterEntities}
            disabled={loadingClusters}
            sx={{
              flex: 1,
              borderRadius: "30px",
              py: 1.2,
              fontSize: "1rem",
              fontWeight: 600,
              textTransform: "none",
              border: "2px solid #5B8DEF",
              color: "#5B8DEF",
              "&:hover": {
                backgroundColor: "#E8F1FF",
                borderColor: "#5B8DEF",
              },
              transition: "all 0.3s ease-in-out",
            }}
          >
            {loadingClusters ? (
              <CircularProgress size={24} sx={{ color: "#5B8DEF" }} />
            ) : (
              "Pinpoint Similar Entities"
            )}
          </Button>

          <Typography
            variant="body1"
            sx={{ fontWeight: 600, color: "grey.600", mx: 1 }}
          >
            OR
          </Typography>

          <TextField
            label="Enter Entity ID"
            variant="outlined"
            value={entityIdInput}
            onChange={(e) => setEntityIdInput(e.target.value)}
            sx={{
              flex: 1,
              "& .MuiOutlinedInput-root": {
                borderRadius: "30px",
              },
            }}
          />

          <Button
            variant="outlined"
            color="secondary"
            onClick={handleFetchEntityById}
            sx={{
              borderRadius: "30px",
              py: 1.2,
              fontSize: "1rem",
              fontWeight: 600,
              textTransform: "none",
              border: "2px solid #5B8DEF",
              color: "#5B8DEF",
              "&:hover": {
                backgroundColor: "#E8F1FF",
                borderColor: "#5B8DEF",
              },
              transition: "all 0.3s ease-in-out",
            }}
          >
            Fetch Entity
          </Button>
        </Box>

        {Object.keys(clusters).length > 0 && (
          <Box>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Scatter Plot of Clusters
            </Typography>
            <ScatterChart
              width={800}
              height={400}
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
              <CartesianGrid />
              <XAxis type="number" dataKey="coordinates[0]" />
              <YAxis type="number" dataKey="coordinates[1]" />
              <Tooltip content={renderTooltip} />
              <Legend
                onClick={(e) => handleLegendClick(e.value)}
                wrapperStyle={{ cursor: "pointer" }}
              />
              {renderScatterPlots()}
            </ScatterChart>
          </Box>
        )}
      </Paper>

      {selectedEntity && (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" sx={{ mb: 2 }}>
            Selected Entity: #{selectedEntity.index}
          </Typography>

          <pre
            style={{
              background: "#f9f9f9",
              padding: "8px",
              borderRadius: "8px",
              overflow: "auto",
            }}
          >
            {JSON.stringify(selectedEntity.features, null, 2)}
          </pre>

          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(e, newMode) => setViewMode(newMode)}
            sx={{ mb: 2 }}
          >
            <ToggleButton value="counterfactual">Counterfactual</ToggleButton>
            <ToggleButton value="recourse">Actionable Recourse</ToggleButton>
          </ToggleButtonGroup>

          {viewMode === "counterfactual" ? (
            <>
              {/* Counterfactual Analysis Section */}
              <Box
                sx={{
                  mb: 4,
                  p: 3,
                  border: "1px solid #E0E0E0",
                  borderRadius: "12px",
                }}
              >
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Counterfactual Analysis
                </Typography>

                <FormControl fullWidth sx={{ mb: 3 }}>
                  <InputLabel shrink={true} sx={{ background: "#fff", px: 1 }}>
                    Select New {sensitiveAttr}
                  </InputLabel>
                  <Select
                    value={newSensitiveValue}
                    onChange={(e) => setNewSensitiveValue(e.target.value)}
                    displayEmpty
                  >
                    {possibleSensitiveValues
                      .filter(
                        (value) =>
                          value !== selectedEntity.features[sensitiveAttr]
                      )
                      .map((value, idx) => (
                        <MenuItem key={idx} value={value}>
                          {value}
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>

                <Box
                  sx={{ display: "flex", justifyContent: "flex-end", mb: 3 }}
                >
                  <Button
                    variant="outlined"
                    color="secondary"
                    onClick={handleGenerateCounterfactual}
                    disabled={loadingCounterfactual}
                    sx={{
                      flex: 1,
                      borderRadius: "30px",
                      py: 1.2,
                      fontSize: "1rem",
                      fontWeight: 600,
                      textTransform: "none",
                      border: "2px solid #5B8DEF",
                      color: "#5B8DEF",
                      "&:hover": {
                        backgroundColor: "#E8F1FF",
                        borderColor: "#5B8DEF",
                      },
                      transition: "all 0.3s ease-in-out",
                    }}
                  >
                    {loadingCounterfactual ? (
                      <CircularProgress size={20} sx={{ color: "#fff" }} />
                    ) : (
                      "Generate Counterfactual"
                    )}
                  </Button>
                </Box>
              </Box>

              {counterfactual && (
                <Paper
                  elevation={3}
                  sx={{
                    mt: 3,
                    p: 3,
                    borderRadius: "12px",
                    background: "#f9f9f9",
                    border: "1px solid #E0E0E0",
                  }}
                >
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: 600,
                      color:
                        counterfactual.original_prediction ===
                        counterfactual.counterfactual_prediction
                          ? "green"
                          : "red",
                      mb: 2,
                    }}
                  >
                    {counterfactual.original_prediction ===
                    counterfactual.counterfactual_prediction
                      ? "Prediction did not change."
                      : "Prediction changed!"}
                  </Typography>

                  <Box sx={{ display: "flex", gap: 4, mb: 3 }}>
                    <Typography variant="body1">
                      <strong>Original Prediction:</strong>{" "}
                      {counterfactual.original_prediction}
                    </Typography>
                    <Typography variant="body1">
                      <strong>Counterfactual Prediction:</strong>{" "}
                      {counterfactual.counterfactual_prediction}
                    </Typography>
                  </Box>

                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Modified Features:</strong>
                  </Typography>
                  <Box
                    component="pre"
                    sx={{
                      background: "#fff",
                      p: 2,
                      borderRadius: "8px",
                      border: "1px solid #E0E0E0",
                      overflow: "auto",
                      fontSize: "0.9rem",
                    }}
                  >
                    {JSON.stringify(counterfactual.flipped_features, null, 2)}
                  </Box>
                </Paper>
              )}
            </>
          ) : (
            <ActionableRecourse
              dataset={dataset}
              targetAttr={targetAttr}
              sensitiveAttr={sensitiveAttr}
              selectedEntity={selectedEntity}
            />
          )}
        </Paper>
      )}
    </Box>
  );
}
