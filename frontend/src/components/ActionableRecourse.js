import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Chip,
  OutlinedInput,
} from "@mui/material";
import axios from "axios";
import { useDataset } from "./DatasetContext";

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

export default function ActionableRecourse({
  dataset,
  targetAttr,
  sensitiveAttr,
  selectedEntity,
}) {
  const [loadingRecourse, setLoadingRecourse] = useState(false);
  const [recourse, setRecourse] = useState(null);
  const [actionableFeatures, setActionableFeatures] = useState([]);
  const [availableFeatures, setAvailableFeatures] = useState([]);
  const { privilegedGroup, unprivilegedGroup } = useDataset();
  const targetPrediction = selectedEntity?.prediction === 0 ? "0" : "1";

  useEffect(() => {
    if (selectedEntity && selectedEntity.features) {
      const allFeatures = Object.keys(selectedEntity.features);
      const filteredFeatures = allFeatures.filter(
        (feature) => feature !== targetAttr && feature !== sensitiveAttr
      );
      setAvailableFeatures(filteredFeatures);
    }
  }, [selectedEntity, targetAttr, sensitiveAttr]);

  const handleGenerateRecourse = async () => {
    if (!selectedEntity) {
      alert("Please select an entity.");
      return;
    }
    if (actionableFeatures.length === 0) {
      alert("Please select at least one actionable feature.");
      return;
    }
    setLoadingRecourse(true);
    setRecourse(null);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/counterfactual/recourse",
        {
          data_path: dataset,
          target_column: targetAttr,
          selected_entity_index: selectedEntity.index,
          actionable_features: actionableFeatures,
          target_prediction: targetPrediction,
          lambda_weight: 0.5,
        }
      );

      setRecourse(response.data.recourse);
    } catch (error) {
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoadingRecourse(false);
    }
  };

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        Actionable Recourse
      </Typography>

      {/* Check if prediction is 0 */}
      {selectedEntity?.prediction === 0 ? (
        <>
          {/* Explanation */}
          <Typography variant="body2" sx={{ color: "grey.600", mb: 3 }}>
            Actionable recourse helps identify the minimal changes needed in
            certain features to achieve a desired outcome. You can select which
            features you are willing to change (e.g., hours worked, education
            level).
          </Typography>

          {/* Select actionable features */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel id="actionable-features-label">
              Select Actionable Features
            </InputLabel>
            <Select
              labelId="actionable-features-label"
              multiple
              value={actionableFeatures}
              onChange={(e) => setActionableFeatures(e.target.value)}
              input={
                <OutlinedInput
                  id="select-multiple-chip"
                  label="Select Actionable Features"
                />
              }
              renderValue={(selected) => (
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                  {selected.map((value) => (
                    <Chip key={value} label={value} />
                  ))}
                </Box>
              )}
              MenuProps={MenuProps}
            >
              {availableFeatures.map((feature) => (
                <MenuItem key={feature} value={feature}>
                  {feature}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 3 }}>
            {/* Generate Recourse Button */}
            <Button
              variant="outlined"
              color="primary"
              onClick={handleGenerateRecourse}
              disabled={loadingRecourse}
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
              {loadingRecourse ? (
                <CircularProgress size={20} sx={{ color: "#fff" }} />
              ) : (
                "Generate Recourse"
              )}
            </Button>
          </Box>
          {/* Display Recourse Result */}
          {recourse ? (
            <Box sx={{ mt: 3 }}>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Suggested changes to achieve the desired outcome:
              </Typography>
              <pre
                style={{
                  background: "#f9f9f9",
                  padding: "8px",
                  borderRadius: "8px",
                  overflow: "auto",
                }}
              >
                {JSON.stringify(recourse, null, 2)}
              </pre>
            </Box>
          ) : (
            !loadingRecourse && (
              <Typography
                variant="body2"
                sx={{ color: "grey.600", mt: 2, fontStyle: "italic" }}
              >
                No recourse generated yet. Select actionable features and click
                "Generate Recourse".
              </Typography>
            )
          )}
        </>
      ) : (
        <Typography
          variant="body2"
          sx={{
            color: "grey.600",
            mt: 2,
            background: "#f9f9f9",
            padding: "10px",
            borderRadius: "8px",
          }}
        >
          Actionable Recourse is only available for entities with an undesired
          outcome.
        </Typography>
      )}
    </Box>
  );
}
