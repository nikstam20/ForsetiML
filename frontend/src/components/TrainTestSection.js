import React from "react";
import {
  Paper,
  Typography,
  TextField,
  Slider,
  Button,
  IconButton,
  Tooltip,
  MenuItem,
} from "@mui/material";
import { Add, Remove, Info } from "@mui/icons-material";
import parameterInfo from "../utils/parameterInfo";

const TrainTestSection = ({
  splitPercentage,
  setSplitPercentage,
  modelParams,
  setModelParams,
}) => {
  const handleSplitChange = (_, value) => setSplitPercentage(value);
  const handleAddParam = () =>
    setModelParams([...modelParams, { key: "", value: "" }]);
  const handleRemoveParam = (index) =>
    setModelParams(modelParams.filter((_, i) => i !== index));
  const handleParamChange = (index, field, value) => {
    const updatedParams = [...modelParams];
    updatedParams[index][field] = value;
    setModelParams(updatedParams);
  };

  return (
    <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Train and Test Model
      </Typography>
      <Slider
        value={splitPercentage}
        onChange={handleSplitChange}
        min={50}
        max={90}
        step={5}
        valueLabelDisplay="auto"
        sx={{ mb: 2 }}
      />
      {modelParams.map((param, index) => (
        <div
          key={index}
          style={{ display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <TextField
            select
            label="Key"
            value={param.key}
            onChange={(e) => handleParamChange(index, "key", e.target.value)}
            sx={{ width: 200 }}
          >
            {Object.keys(parameterInfo).map((key) => (
              <MenuItem key={key} value={key}>
                {key}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            label="Value"
            value={param.value}
            onChange={(e) => handleParamChange(index, "value", e.target.value)}
            sx={{ width: 200 }}
          />
          <Tooltip title={parameterInfo[param.key] || "No description"}>
            <IconButton>
              <Info />
            </IconButton>
          </Tooltip>
          <IconButton onClick={() => handleRemoveParam(index)}>
            <Remove />
          </IconButton>
        </div>
      ))}
      <Button variant="outlined" onClick={handleAddParam} startIcon={<Add />}>
        Add Parameter
      </Button>
    </Paper>
  );
};

export default TrainTestSection;
