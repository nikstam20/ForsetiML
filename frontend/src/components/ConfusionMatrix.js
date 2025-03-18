import React from "react";
import { Box, Typography } from "@mui/material";

const ConfusionMatrix = ({ matrix }) => {
  return (
    <Box sx={{ mb: 4 }}>
      <Typography
        variant="h6"
        sx={{
          fontWeight: "bold",
          mb: 2,
          textAlign: "center",
          color: "#34568B",
        }}
      >
        Confusion Matrix
      </Typography>
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gridTemplateRows: "repeat(3, auto)",
          textAlign: "center",
          backgroundColor: "#F7FAFC",
          borderRadius: 2,
          boxShadow: 3,
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#F1F5F9",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderBottom: "1px solid #E2E8F0",
          }}
        >
          <Typography
            variant="subtitle2"
            sx={{
              writingMode: "vertical-lr",
              transform: "rotate(180deg)",
              fontWeight: "bold",
              color: "#718096",
              fontSize: "0.9rem",
            }}
          >
            Actual
          </Typography>
        </Box>

        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#6C8EBF",
            color: "white",
            p: 1,
            borderBottom: "1px solid #E2E8F0",
          }}
        >
          Negative
        </Box>
        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#6C8EBF",
            color: "white",
            p: 1,
            borderBottom: "1px solid #E2E8F0",
          }}
        >
          Positive
        </Box>

        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#6C8EBF",
            color: "white",
            p: 1,
            borderRight: "1px solid #E2E8F0",
          }}
        >
          Negative
        </Box>
        <Box
          sx={{
            backgroundColor: "#C8E6C9",
            p: 2,
            fontWeight: "bold",
            color: "#2D3748",
          }}
        >
          {matrix.true_negatives}
        </Box>
        <Box
          sx={{
            backgroundColor: "#FFCDD2",
            p: 2,
            fontWeight: "bold",
            color: "#2D3748",
          }}
        >
          {matrix.false_positives}
        </Box>

        <Box
          sx={{
            fontWeight: "bold",
            backgroundColor: "#6C8EBF",
            color: "white",
            p: 1,
            borderRight: "1px solid #E2E8F0",
          }}
        >
          Positive
        </Box>
        <Box
          sx={{
            backgroundColor: "#FFCDD2",
            p: 2,
            fontWeight: "bold",
            color: "#2D3748",
          }}
        >
          {matrix.false_negatives}
        </Box>
        <Box
          sx={{
            backgroundColor: "#C8E6C9",
            p: 2,
            fontWeight: "bold",
            color: "#2D3748",
          }}
        >
          {matrix.true_positives}
        </Box>
      </Box>
      <Typography
        variant="subtitle2"
        sx={{
          textAlign: "center",
          mt: 1,
          fontWeight: "bold",
          color: "#718096",
          fontSize: "0.9rem",
        }}
      >
        Predicted
      </Typography>
    </Box>
  );
};

export default ConfusionMatrix;
