import React from "react";
import {
  Box,
  Typography,
  Container,
  Card,
  CardContent,
  Paper,
  Grid,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";

const theme = createTheme({
  typography: {
    fontFamily: "'Inter', sans-serif",
    h4: { fontWeight: 700, fontSize: "2rem" },
    h5: { fontWeight: 600, fontSize: "1.5rem" },
    subtitle1: { fontWeight: 500, fontSize: "1.2rem" },
    body2: { fontSize: "1rem", color: "#4A5568" },
  },
  palette: {
    primary: { main: "#1565C0" },
    secondary: { main: "#E3F2FD" },
  },
});

const fairnessNotions = {
  Independence: [
    {
      name: "Statistical Parity",
      description:
        "Ensures different groups receive positive outcomes at the same rate, regardless of merit.",
      example:
        "If 60% of Group A gets a loan, 60% of Group B should also be approved.",
    },
  ],
  Separation: [
    {
      name: "Equalized Odds",
      description:
        "Ensures true positive rates and false positive rates are equal across groups.",
      example:
        "A hiring model should not favor one group by approving more unqualified candidates.",
    },
    {
      name: "Equal Opportunity",
      description:
        "Ensures qualified individuals have an equal chance of receiving a positive outcome across groups.",
      example:
        "If 90% of qualified men get promoted, 90% of qualified women should also get promoted.",
    },
    {
      name: "Balance for Positive Class",
      description:
        "Ensures that the model's correct positive predictions are equally reliable across groups.",
      example:
        "A cancer detection model should have the same precision for all ethnic groups.",
    },
    {
      name: "Balance for Negative Class",
      description:
        "Ensures that correct negative predictions are equally reliable across groups.",
      example:
        "A fraud detection system should not falsely accuse one group more than another.",
    },
  ],
  Sufficiency: [
    {
      name: "Predictive Equality",
      description: "Ensures false positive rates are the same across groups.",
      example:
        "If a criminal risk model wrongly classifies 5% of Group A as high risk, it should do the same for Group B.",
    },
    {
      name: "Conditional Use Accuracy Equality",
      description:
        "Ensures both positive and negative predictions are equally accurate across groups.",
      example:
        "If a job screening test has 85% accuracy for Group A, it should also be 85% for Group B.",
    },
    {
      name: "Well Calibration",
      description:
        "Ensures predicted probabilities reflect real-world outcomes consistently across groups.",
      example:
        "A college admission system saying '80% acceptance chance' should actually mean 80% get accepted.",
    },
    {
      name: "Test Fairness",
      description:
        "Measures overall fairness by combining multiple fairness metrics.",
      example:
        "Balances different fairness notions to evaluate model bias holistically.",
    },
  ],
};

function FairnessGlossary() {
  return (
    <ThemeProvider theme={theme}>
      <Container
        maxWidth="lg"
        sx={{
          display: "flex",
          flexDirection: "column",
          flex: 1,
          transform: "scale(0.9)",
          transformOrigin: "top left",
          mt: -2,
          ml: -25,
        }}
      >
        {/* Hero Section */}
        <Paper
          elevation={6}
          sx={{
            background:
              "linear-gradient(to right,rgb(154, 178, 206),rgb(119, 160, 195))",
            padding: "30px 25px",
            mb: 4,
            borderLeft: "6px rgb(6, 8, 56)",
            borderRadius: 2,
          }}
        >
          <Typography
            variant="h4"
            color="rgb(44, 46, 104)"
            sx={{ fontWeight: "bold" }}
          >
            Understanding Fairness in Machine Learning
          </Typography>
          <Typography
            variant="body2"
            sx={{ mt: 1.5, fontSize: "1.1rem", color: "#37474F" }}
          >
            This glossary defines key <b>fairness notions</b> used to evaluate
            bias in AI models.
          </Typography>
          <Typography
            variant="body2"
            sx={{ mt: 2, fontSize: "1rem", color: "#37474F" }}
          >
            Click on any category below to explore its fairness notions,
            explanations, and real-world examples.
          </Typography>
        </Paper>

        {Object.entries(fairnessNotions).map(([category, notions]) => (
          <Box key={category} sx={{ mb: 4 }}>
            {/* Visually Enhanced Section Header */}
            <Paper
              elevation={4}
              sx={{
                backgroundColor: "#E3F2FD",
                padding: "15px 20px",
                borderLeft: "5px solid #1565C0",
                borderRadius: 1,
              }}
            >
              <Typography variant="h5" color="primary">
                {category}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                {category === "Independence" &&
                  "Focuses on ensuring equal treatment across groups, regardless of actual outcomes."}
                {category === "Separation" &&
                  "Ensures fairness by balancing model errors (false positives, false negatives) across groups."}
                {category === "Sufficiency" &&
                  "Ensures predicted probabilities are accurate and consistent across groups."}
              </Typography>
            </Paper>

            {/* Fairness Notions */}
            <Grid container spacing={2} sx={{ mt: 2 }}>
              {notions.map((notion) => (
                <Grid item xs={12} md={6} key={notion.name}>
                  <Card
                    sx={{
                      boxShadow: 3,
                      borderRadius: 2,
                      transition: "0.3s",
                      "&:hover": { transform: "scale(1.03)" },
                    }}
                  >
                    <CardContent>
                      <Typography
                        variant="subtitle1"
                        color="primary"
                        sx={{ fontWeight: "bold" }}
                      >
                        {notion.name}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {notion.description}
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ fontWeight: "bold", mt: 2 }}
                      >
                        Example: {notion.example}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        ))}
      </Container>
    </ThemeProvider>
  );
}

export default FairnessGlossary;
