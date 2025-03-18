import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Container,
  TextField,
  Button,
  Typography,
  Paper,
  Grid2,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Modal,
  Box,
  Card,
  CardContent,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import CytoscapeComponent from "react-cytoscapejs";
import dagre from "cytoscape-dagre";
import cytoscape from "cytoscape";
import CloseIcon from "@mui/icons-material/Close";
import OpenInFullIcon from "@mui/icons-material/OpenInFull";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { RadialChart } from "react-vis";
import { Search } from "@mui/icons-material";
import { motion } from "framer-motion";
import { AnimatePresence } from "framer-motion";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { useDataset } from "./DatasetContext";
import { alignProperty } from "@mui/material/styles/cssUtils";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

cytoscape.use(dagre);
// const theme = createTheme({
//   typography: {
//     fontFamily: "Roboto, Arial, sans-serif",
//     fontWeightRegular: 400,
//     fontWeightBold: 700,
//   },
// });
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    },
  },
};
// Define Sophisticated Blue Color Palette
const theme = createTheme({
  palette: {
    primary: {
      main: "#34568B", // Sophisticated Navy Blue
    },
    secondary: {
      main: "#6C8EBF", // Cool Gray-Blue Accent
    },
    background: {
      default: "#F7FAFC", // Light Grayish-Blue Background
      paper: "#FFFFFF", // Clean White for Boxes
    },
    text: {
      primary: "#2D3748", // Dark Gray for Text
      secondary: "#718096", // Subtle Gray for Secondary Text
    },
  },
  typography: {
    fontFamily: "'Roboto', Arial, sans-serif",
    fontWeightRegular: 400,
    fontWeightBold: 700,
  },
  shape: {
    borderRadius: 8, // Reduced corners for outer components
  },
});

// const theme = createTheme({
//   palette: {
//     primary: {
//       main: "#34568B", // Sophisticated Navy Blue
//     },
//     secondary: {
//       main: "#6C8EBF", // Cool Gray-Blue Accent
//     },
//     background: {
//       default: "#F7FAFC", // Light Grayish-Blue Background
//       paper: "#FFFFFF", // Clean White for Boxes
//     },
//     text: {
//       primary: "#2D3748", // Dark Gray for Text
//       secondary: "#718096", // Subtle Gray for Secondary Text
//     },
//   },
//   typography: {
//     fontFamily: "'Roboto', Arial, sans-serif",
//     fontWeightRegular: 400,
//     fontWeightBold: 700,
//   },
//   shape: {
//     borderRadius: 6, // Reduced corners for outer components
//   },
// });

function DataAnalysis() {
  const [file, setFile] = useState(null);
  const [datasetPreview, setDatasetPreview] = useState([]);
  //  const [attributes, setAttributes] = useState([]);
  //  const [sensitiveAttr, setSensitiveAttr] = useState("");
  //  const [targetAttr, setTargetAttr] = useState("");
  const [staticVars, setStaticVars] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [confounders, setConfounders] = useState([]);
  const [mediators, setMediators] = useState([]);
  const [mediatorPaths, setMediatorPaths] = useState([]);
  const [legitimateVars, setLegitimateVars] = useState([]);
  const [resolvingVars, setResolvingVars] = useState([]);
  const [proxyVars, setProxyVars] = useState([]);
  const [pathSpecificFairness, setPathSpecificFairness] = useState(null);
  const [noUnresolvedDiscrimination, setNoUnresolvedDiscrimination] =
    useState(null);
  const [unresolvedDiscriminationPaths, setUnresolvedDiscriminationPaths] =
    useState([]);
  const [noProxyDiscrimination, setNoProxyDiscrimination] = useState(null);
  const [proxyDiscriminationPaths, setProxyDiscriminationPaths] = useState([]);
  const [groupMetrics, setGroupMetrics] = useState([]);
  const [loadingGroupMetrics, setLoadingGroupMetrics] = useState(false);
  // const [privilegedGroup, setPrivilegedGroup] = useState(null);
  // const [unprivilegedGroup, setUnprivilegedGroup] = useState(null);
  const [sensitiveAttrValues, setSensitiveAttrValues] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [visGraph, setVisGraph] = useState(false);
  const [pdfData, setPdfData] = useState(null);
  const [loadingPDF, setLoadingPDF] = useState(false);
  const [samplingParity, setSamplingParity] = useState({
    privileged: 0,
    unprivileged: 0,
  });
  const {
    dataset,
    setDataset,
    attributes,
    setAttributes,
    sensitiveAttr,
    setSensitiveAttr,
    targetAttr,
    setTargetAttr,
    privilegedGroup,
    setPrivilegedGroup,
    unprivilegedGroup,
    setUnprivilegedGroup,
  } = useDataset();
  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    console.log("File selected:", uploadedFile);
  };

  const handleFileUpload = async () => {
    if (!file) {
      alert("Please upload a file first!");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("Uploading file...");

      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData
      );
      setDataset(response.data.path.replace(/\\/g, "/"));
      console.log(dataset);
      const datasetResponse = await axios.get(
        `http://127.0.0.1:5000/api/preview-dataset?path=${response.data.path.replace(
          /\\/g,
          "/"
        )}`
      );
      let parsedData = datasetResponse.data;

      if (typeof parsedData === "string") {
        parsedData = JSON.parse(parsedData.replace(/NaN/g, "null"));
      } else {
        parsedData = JSON.parse(
          JSON.stringify(parsedData, (_, value) =>
            value === "NaN" ? null : value
          )
        );
      }
      setDatasetPreview(parsedData.preview || []);
      setAttributes(parsedData.columns || []);
    } catch (error) {
      console.error("File upload failed:", error);
      alert("Failed to upload the file or fetch dataset preview.");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateGraph = async () => {
    if (!sensitiveAttr || !targetAttr || staticVars.length === 0) {
      alert("Please select all attributes!");
      return;
    }

    setLoadingGraph(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/causal/create_causal_graph",
        {
          data_path: dataset,
          sensitive_attr: sensitiveAttr,
          target_attr: targetAttr,
          static_vars: staticVars,
        }
      );
      let graph = response.data.graph;
      setGraphData({
        nodes: graph.nodes || [],
        links: (graph.edges || []).map((edge) => ({
          source: edge.source,
          target: edge.target,
          weight: edge.weight,
        })),
      });
      const mediatorResponse = await axios.post(
        "http://127.0.0.1:5000/api/causal/mediators",
        {
          sensitive_attr: sensitiveAttr,
          target_attr: targetAttr,
          edges: graph.edges,
        }
      );

      setMediators(mediatorResponse.data.mediators || []);
      setMediatorPaths(mediatorResponse.data.mediator_paths || []);
      try {
        const response = await axios.post(
          "http://127.0.0.1:5000/api/causal/confounders",
          {
            sensitive_attr: sensitiveAttr,
            target_attr: targetAttr,
            nodes: graphData.nodes,
            edges: graphData.links,
          }
        );
        setConfounders(response.data.confounders || []);
      } catch (error) {
        console.error("Failed to fetch confounders:", error);
        alert("Error fetching confounders.");
      }
    } catch (error) {
      console.error("Failed to create causal graph:", error);
      alert("Error creating causal graph.");
    } finally {
      setLoadingGraph(false);
      setVisGraph(true);
    }
  };

  const handleFetchConfounders = async () => {
    if (!sensitiveAttr || !targetAttr) {
      alert(
        "Sensitive and Target attributes must be selected to fetch confounders!"
      );
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/causal/confounders",
        {
          sensitive_attr: sensitiveAttr,
          target_attr: targetAttr,
          nodes: graphData.nodes,
          edges: graphData.links,
        }
      );
      setConfounders(response.data.confounders || []);
    } catch (error) {
      console.error("Failed to fetch confounders:", error);
      alert("Error fetching confounders.");
    } finally {
      setLoading(false);
    }
  };

  const handlePathSpecificFairness = async () => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/causal/path_specific_fairness",
        {
          sensitive_attr: sensitiveAttr,
          target_attr: targetAttr,
          legitimate_vars: legitimateVars,
          proxy_vars: proxyVars,
          nodes: graphData.nodes,
          edges: graphData.links,
        }
      );
      setPathSpecificFairness(response.data);
    } catch (error) {
      console.error("Failed to update path-specific fairness:", error);
    }
  };

  const handleNoUnresolvedDiscrimination = async (resolvingVars) => {
    console.log(resolvingVars);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/causal/no_unresolved_discrimination",
        {
          sensitive_attr: sensitiveAttr,
          target_attr: targetAttr,
          resolving_vars: resolvingVars,
          nodes: graphData.nodes,
          edges: graphData.links,
        }
      );
      console.log(response.data);
      const result = response.data;

      if (result.resolved.resolved === false) {
        console.log(
          "Unresolved discrimination exists. Failing paths:",
          result.resolved.failing_paths
        );
        setNoUnresolvedDiscrimination(false);
        setUnresolvedDiscriminationPaths(result.resolved.failing_paths || []);
      } else {
        console.log("All paths resolved.");
        setNoUnresolvedDiscrimination(true);
        setUnresolvedDiscriminationPaths([]);
      }
    } catch (error) {
      console.error("Error running No Unresolved Discrimination:", error);
    }
  };

  useEffect(() => {
    handleNoUnresolvedDiscrimination(resolvingVars);
  }, [resolvingVars]);

  const handleNoProxyDiscrimination = async (proxyVars) => {
    console.log(proxyVars);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/causal/no_proxy_discrimination",
        {
          sensitive_attr: sensitiveAttr,
          target_attr: targetAttr,
          proxy_vars: proxyVars,
          nodes: graphData.nodes,
          edges: graphData.links,
        }
      );
      console.log(response.data);
      const result = response.data;

      if (result.no_proxies.no_proxies === false) {
        console.log(
          "Proxy discrimination exists. Failing paths:",
          result.no_proxies.failing_paths
        );
        setNoProxyDiscrimination(false);
      } else {
        console.log("No proxy discrimination detected.");
        setNoProxyDiscrimination(true);
      }

      setProxyDiscriminationPaths(result.no_proxies.failing_paths || []);
    } catch (error) {
      console.error("Error running No Proxy Discrimination:", error);
    }
  };

  useEffect(() => {
    handleNoProxyDiscrimination(proxyVars);
  }, [proxyVars]);

  const fetchGroupMetrics = async () => {
    if (
      !sensitiveAttr ||
      !targetAttr ||
      datasetPreview.length === 0 ||
      privilegedGroup === null ||
      unprivilegedGroup === null
    ) {
      alert(
        "Sensitive attribute, target attribute, privileged group, and unprivileged group are required!"
      );
      return;
    }

    setLoadingGroupMetrics(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/group/metrics",
        {
          data: dataset,
          sensitive_attr: sensitiveAttr,
          target_attr: targetAttr,
          privileged_group: privilegedGroup,
          unprivileged_group: unprivilegedGroup,
        }
      );

      if (response.data && response.data.metrics) {
        const parsedMetrics = JSON.parse(
          response.data.metrics
            .replace(/'/g, '"')
            .replace(/np\.\w+\(([^)]+)\)/g, "$1")
        );

        const privilegedSamplingPercentage = parseFloat(
          parsedMetrics["Sampling Parity Privileged"] || 0
        );
        const unprivilegedSamplingPercentage = parseFloat(
          parsedMetrics["Sampling Parity Unprivileged"] || 0
        );

        setSamplingParity({
          privileged: privilegedSamplingPercentage,
          unprivileged: unprivilegedSamplingPercentage,
        });

        const metricsArray = Object.keys(parsedMetrics)
          .filter(
            (key) =>
              key !== "Sampling Parity Privileged" &&
              key !== "Sampling Parity Unprivileged"
          )
          .map((key) => {
            let metricDetails = {};

            switch (key) {
              case "Disparate Impact":
                metricDetails = {
                  best: 1,
                  worst: 0,
                  min: 0,
                  max: 3,
                };
                break;

              case "Group Mean Difference":
                metricDetails = {
                  best: 0,
                  worst: 1,
                  min: -1,
                  max: 1,
                };
                break;

              case "Mutual Information":
                metricDetails = {
                  best: 0,
                  worst: 1,
                  min: 0,
                  max: 2,
                };
                break;

              case "Consistency":
                metricDetails = {
                  best: 1,
                  worst: 0,
                  min: 0,
                  max: 1,
                };
                break;

              case "Coefficient of Variation":
                metricDetails = {
                  best: 0,
                  worst: 1,
                  min: 0,
                  max: 1.5,
                };
                break;

              default:
                metricDetails = {
                  best: 1,
                  worst: 0,
                  min: 0,
                  max: 1,
                };
                break;
            }

            return {
              name: key,
              value: parseFloat(parsedMetrics[key]),
              ...metricDetails,
            };
          });

        setGroupMetrics(metricsArray);
      } else {
        console.error("Unexpected response format:", response.data);
        setGroupMetrics([]);
      }
    } catch (error) {
      console.error("Failed to fetch group metrics:", error);
      alert("Error fetching group metrics.");
      setGroupMetrics([]);
    } finally {
      setLoadingGroupMetrics(false);
    }
  };

  const fetchPDFData = async () => {
    if (!sensitiveAttr || !targetAttr) {
      alert("Sensitive attribute and target attribute are required!");
      return;
    }

    setLoadingPDF(true);

    try {
      const response = await axios.post("http://127.0.0.1:5000/api/pdf", {
        data: dataset,
        sensitive_attr: sensitiveAttr,
        target_attr: targetAttr,
      });

      setPdfData(response.data);
    } catch (error) {
      console.error("Failed to fetch PDF data:", error);
      alert("Error fetching PDF data.");
      setPdfData(null);
    } finally {
      setLoadingPDF(false);
    }
  };

  useEffect(() => {
    if (sensitiveAttr) {
      const uniqueValues = new Set(
        datasetPreview
          .map((row) => row[sensitiveAttr])
          .filter((val) => val !== null)
      );
      setSensitiveAttrValues(Array.from(uniqueValues));
    } else {
      setSensitiveAttrValues([]);
    }
  }, [sensitiveAttr, datasetPreview]);

  const handlePrivilegedChange = (event) => {
    setPrivilegedGroup(event.target.value);
  };

  const handleUnprivilegedChange = (event) => {
    setUnprivilegedGroup(event.target.value);
  };

  return (
    <ThemeProvider theme={theme}>
      <Container
        maxWidth="lg"
        sx={{
          display: "flex",
          flexDirection: "column",
          flex: 1,
          transform: "scale(0.7)",
          transformOrigin: "top left",
          mt: -3,
          ml: 0,
        }}
      >
        <>
          {/*
   <Box sx={{ display: 'flex', alignItems: 'left', maringLeft: "-1400px" }}>
  <Typography
    variant="h2"
    gutterBottom
    sx={{
      fontWeight: "bold", // Emphasized font weight for the title
      color: theme.palette.primary.main, // Primary color from theme
      fontFamily: "'Roboto', sans-serif", // Modern sans-serif font family
      letterSpacing: "0.5px", // Slightly increased letter spacing for readability
      fontSize: {
        xs: "2.5rem", // Small screens
        sm: "2rem",   // Medium screens
        md: "2rem",   // Large screens
      },
    }}
  >
    Data Analysis:
  </Typography>


  <Typography
    variant="body1"
    sx={{
      color: theme.palette.text.secondary, // Secondary text color for the subheading
      fontSize: {
        xs: "1rem",   // Smaller font on mobile
        sm: "1.25rem", // Medium screens
        md: "1.5rem",  // Larger screens
      },
      fontWeight: "normal", // Regular weight for subheading
    }}
  >
     Explore your data insights with powerful visualizations
  </Typography>
</Box> */}
        </>

        <Box
          sx={{
            display: "grid",
            columnGap: 5,
            rowGap: 5,
            gridTemplateColumns: "1fr 2fr 1fr",
            width: "1450px",
            gridAutoRows: "min-content",
            marginLeft: "-400px",
            border: "1px solid white",
            borderColor: "white",
            "& > *": {
              border: "1px solid white",
            },
          }}
        >
          {/* File Upload Section */}
          <motion.div
            variants={fadeInUp}
            initial="hidden"
            animate="visible"
            style={{ gridColumn: "1 / 2", gridRow: "1 / 2" }}
          >
            <Paper
              elevation={3}
              sx={{
                p: 4,
                borderRadius: theme.shape.borderRadius,
                backgroundColor: theme.palette.background.paper,
                width: "400px", // Take full column width
                gridColumn: "1 / 2", // Span only the first column
                gridRow: "1 / 2", // First row
              }}
            >
              <Typography variant="h5" gutterBottom>
                Upload Dataset
              </Typography>
              <Grid2 container spacing={2} alignItems="center">
                <Grid2 item xs={12} sm={8}>
                  <TextField
                    type="file"
                    fullWidth
                    inputProps={{ accept: ".csv" }}
                    onChange={handleFileChange}
                    label="Choose CSV File"
                    InputLabelProps={{ shrink: true }}
                    variant="outlined"
                  />
                </Grid2>
                <Grid2 item xs={12} sm={4}>
                  <Grid2 container justifyContent="flex-end">
                    <Grid2 item>
                      <Button
                        variant="contained"
                        color="secondary"
                        sx={{
                          borderRadius: "25px",
                          py: 1.5,
                          height: "50px",
                          width: "100px",
                          fontSize: "1rem",
                          fontWeight: 600,
                          textTransform: "none",
                          boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
                          "&:hover": {
                            backgroundColor: "#2D4F82",
                            boxShadow: "0 6px 15px rgba(0, 0, 0, 0.2)",
                          },
                          transition: "all 0.3s ease-in-out",
                          marginLeft: "240px",
                        }}
                        onClick={handleFileUpload}
                        disabled={loading}
                      >
                        Upload
                      </Button>
                    </Grid2>
                  </Grid2>
                </Grid2>
              </Grid2>
            </Paper>
          </motion.div>

          {/* Select Attributes Section */}
          <AnimatePresence>
            {datasetPreview.length > 0 ? (
              <motion.div
                key="attribute-selection"
                variants={fadeInUp}
                initial="hidden"
                animate="visible"
                exit={{ opacity: 0, y: 20 }}
                style={{ gridColumn: "1 / 2", gridRow: "2 / 3" }}
              >
                <Paper
                  elevation={3}
                  sx={{
                    p: 4,
                    borderRadius: theme.shape.borderRadius,
                    backgroundColor: theme.palette.background.paper,
                    width: "400px", // Take full column width
                    gridColumn: "1 / 2", // Span only the first column
                    gridRow: "2 / 3", // Second row
                    alignItems: "start",
                  }}
                >
                  <Typography variant="h5" gutterBottom>
                    Select Attributes
                  </Typography>
                  <Grid2 container spacing={2} direction="column">
                    <Grid2 item xs={12}>
                      <FormControl fullWidth variant="outlined">
                        <InputLabel id="sensitive-attr-label">
                          Sensitive Attribute
                        </InputLabel>
                        <Select
                          labelId="sensitive-attr-label"
                          value={sensitiveAttr}
                          onChange={(e) => setSensitiveAttr(e.target.value)}
                          label="Sensitive Attribute"
                        >
                          {attributes.map((attr, idx) => (
                            <MenuItem key={idx} value={attr}>
                              {attr}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid2>
                    <Grid2 item xs={12}>
                      <FormControl fullWidth variant="outlined">
                        <InputLabel id="target-attr-label">
                          Target Attribute
                        </InputLabel>
                        <Select
                          labelId="target-attr-label"
                          value={targetAttr}
                          onChange={(e) => setTargetAttr(e.target.value)}
                          label="Target Attribute"
                        >
                          {attributes.map((attr, idx) => (
                            <MenuItem key={idx} value={attr}>
                              {attr}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid2>
                    <Grid2 item xs={12}>
                      <FormControl fullWidth variant="outlined">
                        <InputLabel id="static-vars-label">
                          Static Variables
                        </InputLabel>
                        <Select
                          labelId="static-vars-label"
                          multiple
                          value={staticVars}
                          onChange={(e) => setStaticVars(e.target.value)}
                          label="Static Variables"
                        >
                          {attributes.map((attr, idx) => (
                            <MenuItem key={idx} value={attr}>
                              {attr}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid2>
                  </Grid2>
                  <Button
                    variant="contained"
                    color="secondary"
                    size="large"
                    onClick={handleCreateGraph}
                    fullWidth
                    sx={{
                      borderRadius: "25px",
                      mt: 3,
                      py: 1.5,
                      fontSize: "1rem",
                      fontWeight: 600,
                      textTransform: "none",
                      boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
                      "&:hover": {
                        backgroundColor: "#2D4F82",
                        boxShadow: "0 6px 15px rgba(0, 0, 0, 0.2)",
                      },
                      transition: "all 0.3s ease-in-out",
                    }}
                    disabled={loadingGraph}
                  >
                    {loadingGraph ? (
                      <CircularProgress size={24} />
                    ) : (
                      "Compute and Analyse Causal Graph"
                    )}
                  </Button>
                </Paper>
              </motion.div>
            ) : (
              <Typography variant="body2" align="center"></Typography>
            )}
          </AnimatePresence>

          {/* Dataset Preview Section */}
          <AnimatePresence>
            {datasetPreview.length > 0 ? (
              <motion.div
                key="dataset-preview"
                variants={fadeInUp}
                initial="hidden"
                animate="visible"
                exit={{ opacity: 0, y: 20 }}
                style={{ gridColumn: "2 / 3", gridRow: "1 / 3" }}
              >
                <Paper
                  elevation={3}
                  sx={{
                    p: 4,
                    borderRadius: theme.shape.borderRadius,
                    backgroundColor: theme.palette.background.paper,
                    width: "900px",
                    gridColumn: "2 / 3",
                    gridRow: "1 / 3",
                  }}
                >
                  <Typography variant="h5" gutterBottom>
                    Dataset Preview
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          {attributes.map((colName, idx) => (
                            <TableCell
                              key={idx}
                              sx={{
                                fontWeight: "bold", // Make first row bold
                              }}
                            >
                              {colName}
                            </TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {datasetPreview.slice(0, 5).map((row, rowIndex) => (
                          <TableRow key={rowIndex}>
                            {attributes.map((attr, cellIndex) => (
                              <TableCell key={cellIndex}>
                                {row[attr] !== null && row[attr] !== undefined
                                  ? row[attr]
                                  : "-"}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Paper>
              </motion.div>
            ) : (
              <Typography variant="body2" align="center"></Typography>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {datasetPreview.length > 0 && (
              <motion.div
                key="group-metrics"
                variants={fadeInUp}
                initial="hidden"
                animate="visible"
                exit={{ opacity: 0, y: 20 }}
                style={{ gridColumn: "3 / 5", gridRow: "1 / 5" }}
              >
                <Paper
                  elevation={3}
                  sx={{
                    p: 4,
                    borderRadius: theme.shape.borderRadius,
                    backgroundColor: theme.palette.background.paper,
                    width: "600px",
                    align: "start",
                    alignSelf: "start",
                    gridColumn: "3 / 4",
                    gridRow: "1 / 5",
                  }}
                >
                  <Typography variant="h5" gutterBottom>
                    Group Fairness Metrics Dashboard
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    Analyze fairness metrics and compare them across privileged
                    and unprivileged groups.
                  </Typography>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="h6">
                        Privileged/Unprivileged Group Selection
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="h6" sx={{ mb: 2 }}>
                        Select the Unprivileged Group:
                      </Typography>
                      <Box
                        sx={{
                          display: "flex",
                          flexDirection: "column",
                          gap: 2,
                        }}
                      >
                        {sensitiveAttrValues.map((value, idx) => (
                          <Box
                            key={idx}
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              padding: 1,
                              borderRadius: 4,
                              border: "2px solid",
                              borderColor:
                                unprivilegedGroup === value
                                  ? "#3c8dbc"
                                  : "rgba(0, 0, 0, 0.3)",
                              backgroundColor:
                                unprivilegedGroup === value
                                  ? "#e8f4fd"
                                  : "transparent",
                              cursor: "pointer",
                              "&:hover": {
                                borderColor: "#3c8dbc",
                              },
                            }}
                            onClick={() => {
                              setUnprivilegedGroup(value);
                              setPrivilegedGroup(
                                sensitiveAttrValues.find((v) => v !== value)
                              );
                            }}
                          >
                            <input
                              type="radio"
                              checked={unprivilegedGroup === value}
                              onChange={() => {}}
                              style={{ marginRight: 8 }}
                            />
                            <Typography
                              variant="body1"
                              sx={{ fontWeight: "bold" }}
                            >
                              {value}
                            </Typography>
                          </Box>
                        ))}
                      </Box>

                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ mt: 2, display: "block" }}
                      >
                        Once an unprivileged group is selected, the other group
                        will automatically be privileged.
                      </Typography>

                      <Button
                        variant="contained"
                        color="secondary"
                        fullWidth
                        size="large"
                        sx={{
                          borderRadius: "25px",
                          mt: 3,
                          py: 1.5,
                          fontSize: "1rem",
                          fontWeight: 600,
                          textTransform: "none",
                          boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
                          "&:hover": {
                            backgroundColor: "#2D4F82",
                            boxShadow: "0 6px 15px rgba(0, 0, 0, 0.2)",
                          },
                          transition: "all 0.3s ease-in-out",
                        }}
                        onClick={fetchGroupMetrics}
                        disabled={
                          loadingGroupMetrics ||
                          sensitiveAttrValues.length === 0
                        }
                      >
                        {loadingGroupMetrics ? (
                          <CircularProgress size={24} />
                        ) : (
                          "Analyze Group Metrics"
                        )}
                      </Button>
                    </AccordionDetails>
                  </Accordion>
                  {groupMetrics.length > 0 && (
                    <>
                      {/* Statistical Parity Section */}
                      <Typography
                        variant="h6"
                        align="center"
                        sx={{
                          mt: 3,
                          mb: 2,
                          fontWeight: "bold",
                          color: "#3c8dbc",
                        }}
                      >
                        Statistical Parity
                      </Typography>
                      <Grid2
                        container
                        spacing={3}
                        sx={{ mt: 2 }}
                        justifyContent="center"
                      >
                        {groupMetrics
                          .filter((metric) =>
                            [
                              "Disparate Impact",
                              "Group Mean Difference",
                            ].includes(metric.name)
                          )
                          .map((metric, index) => (
                            <Grid2
                              item
                              xs={12}
                              sm={6}
                              md={4}
                              lg={3}
                              key={index}
                              sx={{
                                display: "flex",
                                justifyContent: "center",
                              }}
                            >
                              <Card
                                elevation={3}
                                sx={{
                                  p: 2,
                                  width: "100%",
                                  maxWidth: "280px",
                                  transition: "transform 0.3s, box-shadow 0.3s",
                                  "&:hover": {
                                    transform: "translateY(-5px)",
                                    boxShadow: "0px 5px 15px rgba(0,0,0,0.2)",
                                  },
                                }}
                              >
                                <CardContent>
                                  <Typography
                                    variant="h6"
                                    align="center"
                                    sx={{ fontWeight: "bold" }}
                                  >
                                    {metric.name}
                                  </Typography>
                                  <Typography
                                    variant="body2"
                                    align="center"
                                    sx={{ mt: 1 }}
                                  >
                                    <strong>Value:</strong>{" "}
                                    {metric.value.toFixed(4)}
                                  </Typography>
                                  <Box
                                    sx={{
                                      position: "relative",
                                      height: 20,
                                      backgroundColor: "#f0f0f0",
                                      borderRadius: 5,
                                      mt: 2,
                                      display: "flex",
                                      alignItems: "center",
                                    }}
                                  >
                                    <Box
                                      sx={{
                                        position: "absolute",
                                        left: "5%",
                                        transform: "translateX(-50%)",
                                        top: 0,
                                        bottom: 0,
                                        width: 20,
                                        height: 20,
                                        backgroundColor:
                                          "rgba(255, 99, 71, 0.8)",
                                        borderRadius: "50%",
                                      }}
                                    />
                                    <Typography
                                      sx={{
                                        position: "absolute",
                                        left: "0%",
                                        transform:
                                          "translateX(-50%) translateY(40px)",
                                        fontSize: "12px",
                                      }}
                                    >
                                      {metric.worst.toFixed(2)}
                                    </Typography>

                                    <Box
                                      sx={{
                                        position: "absolute",
                                        left: "95%",
                                        transform: "translateX(-50%)",
                                        top: 0,
                                        bottom: 0,
                                        width: 20,
                                        height: 20,
                                        backgroundColor:
                                          "rgba(144, 238, 144, 0.8)",
                                        borderRadius: "50%",
                                      }}
                                    />
                                    <Typography
                                      sx={{
                                        position: "absolute",
                                        left: "100%",
                                        transform:
                                          "translateX(-50%) translateY(40px)",
                                        fontSize: "12px",
                                      }}
                                    >
                                      {metric.best.toFixed(2)}
                                    </Typography>

                                    <Box
                                      sx={{
                                        position: "absolute",
                                        left: `${Math.min(
                                          100,
                                          ((metric.value - metric.worst) /
                                            (metric.best - metric.worst)) *
                                            100
                                        )}%`,
                                        transform: "translateX(-50%)",
                                        top: -10,
                                        bottom: 0,
                                        width: 5,
                                        height: 40,
                                        backgroundColor: "#f0d343",
                                        borderRadius: "100%",
                                      }}
                                    />
                                    <Typography
                                      sx={{
                                        position: "absolute",
                                        left: `${Math.min(
                                          100,
                                          ((metric.value - metric.worst) /
                                            (metric.best - metric.worst)) *
                                            100
                                        )}%`,
                                        transform:
                                          "translateX(-50%) translateY(40px)",
                                        fontSize: "12px",
                                      }}
                                    >
                                      {metric.value.toFixed(2)}
                                    </Typography>
                                  </Box>
                                </CardContent>
                              </Card>
                            </Grid2>
                          ))}
                      </Grid2>

                      <Typography
                        variant="h6"
                        align="center"
                        sx={{
                          mt: 5,
                          mb: 2,
                          fontWeight: "bold",
                          color: "#3c8dbc",
                        }}
                      >
                        Sufficiency of Data
                      </Typography>
                      <Grid2
                        container
                        spacing={3}
                        sx={{ mt: 2 }}
                        justifyContent="center"
                      >
                        {groupMetrics
                          .filter((metric) =>
                            [
                              "Consistency",
                              "Coefficient of Variation",
                            ].includes(metric.name)
                          )
                          .map((metric, index) => (
                            <Grid2
                              item
                              xs={12}
                              sm={6}
                              md={4}
                              lg={3}
                              key={index}
                              sx={{
                                display: "flex",
                                justifyContent: "center",
                              }}
                            >
                              <Card
                                elevation={3}
                                sx={{
                                  p: 2,
                                  width: "100%",
                                  maxWidth: "280px",
                                  transition: "transform 0.3s, box-shadow 0.3s",
                                  "&:hover": {
                                    transform: "translateY(-5px)",
                                    boxShadow: "0px 5px 15px rgba(0,0,0,0.2)",
                                  },
                                }}
                              >
                                <CardContent>
                                  <Typography
                                    variant="h6"
                                    align="center"
                                    sx={{ fontWeight: "bold" }}
                                  >
                                    {metric.name}
                                  </Typography>
                                  <Typography
                                    variant="body2"
                                    align="center"
                                    sx={{ mt: 1 }}
                                  >
                                    <strong>Value:</strong>{" "}
                                    {metric.value.toFixed(4)}
                                  </Typography>
                                  <Box
                                    sx={{
                                      position: "relative",
                                      height: 20,
                                      backgroundColor: "#f0f0f0",
                                      borderRadius: 5,
                                      mt: 2,
                                      display: "flex",
                                      alignItems: "center",
                                    }}
                                  >
                                    <Box
                                      sx={{
                                        position: "absolute",
                                        left: "5%",
                                        transform: "translateX(-50%)",
                                        top: 0,
                                        bottom: 0,
                                        width: 20,
                                        height: 20,
                                        backgroundColor:
                                          "rgba(255, 99, 71, 0.8)",
                                        borderRadius: "50%",
                                      }}
                                    />
                                    <Typography
                                      sx={{
                                        position: "absolute",
                                        left: "0%",
                                        transform:
                                          "translateX(-50%) translateY(40px)",
                                        fontSize: "12px",
                                      }}
                                    >
                                      {metric.worst.toFixed(2)}
                                    </Typography>

                                    <Box
                                      sx={{
                                        position: "absolute",
                                        left: "95%",
                                        transform: "translateX(-50%)",
                                        top: 0,
                                        bottom: 0,
                                        width: 20,
                                        height: 20,
                                        backgroundColor:
                                          "rgba(144, 238, 144, 0.8)",
                                        borderRadius: "50%",
                                      }}
                                    />
                                    <Typography
                                      sx={{
                                        position: "absolute",
                                        left: "100%",
                                        transform:
                                          "translateX(-50%) translateY(40px)",
                                        fontSize: "12px",
                                      }}
                                    >
                                      {metric.best.toFixed(2)}
                                    </Typography>

                                    <Box
                                      sx={{
                                        position: "absolute",
                                        left: `${Math.min(
                                          100,
                                          ((metric.value - metric.worst) /
                                            (metric.best - metric.worst)) *
                                            100
                                        )}%`,
                                        transform: "translateX(-50%)",
                                        top: -10,
                                        bottom: 0,
                                        width: 5,
                                        height: 40,
                                        backgroundColor: "#f0d343",
                                        borderRadius: "100%",
                                      }}
                                    />

                                    <Typography
                                      sx={{
                                        position: "absolute",
                                        left: `${Math.min(
                                          100,
                                          ((metric.value - metric.worst) /
                                            (metric.best - metric.worst)) *
                                            100
                                        )}%`,
                                        transform:
                                          "translateX(-50%) translateY(40px)",
                                        fontSize: "12px",
                                      }}
                                    >
                                      {metric.value.toFixed(2)}
                                    </Typography>
                                  </Box>
                                </CardContent>
                              </Card>
                            </Grid2>
                          ))}
                      </Grid2>
                      <Typography
                        variant="h6"
                        align="center"
                        sx={{
                          mt: 5,
                          mb: 2,
                          fontWeight: "bold",
                          color: "#3c8dbc",
                        }}
                      >
                        Sampling Parity
                      </Typography>
                      <Grid2
                        container
                        spacing={3}
                        sx={{ mt: 2 }}
                        justifyContent="center"
                      >
                        <Grid2
                          item
                          xs={12}
                          sm={6}
                          md={4}
                          lg={3}
                          sx={{
                            display: "flex",
                            justifyContent: "center",
                          }}
                        >
                          <Card
                            elevation={3}
                            sx={{
                              p: 2,
                              width: "100%",
                              maxWidth: "280px",
                              transition: "transform 0.3s, box-shadow 0.3s",
                              "&:hover": {
                                transform: "translateY(-5px)",
                                boxShadow: "0px 5px 15px rgba(0,0,0,0.2)",
                              },
                            }}
                          >
                            <CardContent>
                              <RadialChart
                                data={[
                                  {
                                    angle:
                                      (samplingParity?.privileged || 0) * 100,
                                    label: `${privilegedGroup}: ${(
                                      (samplingParity?.privileged || 0) * 100
                                    ).toFixed(1)}%`,
                                    color: "#f0d343",
                                  },
                                  {
                                    angle:
                                      (samplingParity?.unprivileged || 0) * 100,
                                    label: `${unprivilegedGroup}: ${(
                                      (samplingParity?.unprivileged || 0) * 100
                                    ).toFixed(1)}%`,
                                    color: "#FAFAD2",
                                  },
                                ]}
                                width={250}
                                height={250}
                                showLabels
                                colorType="literal"
                                labelsAboveChildren
                                animation
                                innerRadius={10}
                                radius={100}
                                padAngle={0.08}
                                style={{
                                  labels: { fontSize: 16, fontWeight: "bold" },
                                }}
                              />
                              <Typography
                                variant="body2"
                                align="center"
                                sx={{ mt: 2, fontSize: "14px", color: "gray" }}
                              >
                                Represents the percentage of samples for each
                                group.
                              </Typography>
                            </CardContent>
                          </Card>
                        </Grid2>
                      </Grid2>
                    </>
                  )}
                </Paper>
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence>
            {visGraph ? (
              <motion.div
                key="group-metrics"
                variants={fadeInUp}
                initial="hidden"
                animate="visible"
                exit={{ opacity: 0, y: 20 }}
                style={{ gridColumn: "1 / 3", gridRow: "3 / 4" }}
              >
                <Paper
                  elevation={3}
                  sx={{
                    p: 4,
                    borderRadius: theme.shape.borderRadius,
                    backgroundColor: theme.palette.background.paper,
                    // width: "1200px",
                    align: "start",
                    alignSelf: "start",
                    gridColumn: "1 / 3",
                    gridRow: "3 / 4",
                  }}
                >
                  <Typography
                    variant="h5"
                    gutterBottom
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    Causal Graph
                    <IconButton onClick={() => setIsMaximized(true)}>
                      <OpenInFullIcon />
                    </IconButton>
                  </Typography>
                  <div
                    style={{ display: "flex", height: "500px", width: "100%" }}
                  >
                    <div style={{ flex: 4 }}>
                      <CytoscapeComponent
                        elements={[
                          {
                            data: { id: sensitiveAttr, label: sensitiveAttr },
                            position: { x: 100, y: 250 },
                          },
                          {
                            data: { id: targetAttr, label: targetAttr },
                            position: { x: 900, y: 250 },
                          },
                          ...graphData.nodes
                            .filter(
                              (node) =>
                                node.id !== sensitiveAttr &&
                                node.id !== targetAttr
                            )
                            .map((node, index, array) => ({
                              data: {
                                id: node.id,
                                label: node.id,
                                isConfounder: confounders.includes(node.id),
                              },
                              position: {
                                x: 200 + (index * 700) / array.length,
                                y: 100 + (index % 5) * 80,
                              },
                            })),
                          ...graphData.links.map((link) => ({
                            data: {
                              source: link.source,
                              target: link.target,
                              weight: link.weight,
                              label: `Weight: ${link.weight.toFixed(2)}`,
                            },
                          })),
                        ]}
                        style={{
                          width: "100%",
                          height: "100%",
                        }}
                        layout={{
                          name: "preset",
                        }}
                        stylesheet={[
                          {
                            selector: "node",
                            style: {
                              "background-color": (node) =>
                                node.data("id") === sensitiveAttr
                                  ? "#FAFAD2"
                                  : node.data("id") === targetAttr
                                  ? "#87CEFA"
                                  : "#DCDCDC",
                              "border-width": 2,
                              "border-color": "black",
                              "border-style": (node) =>
                                node.data("isConfounder") ? "dashed" : "solid",
                              label: "data(label)",
                              "text-valign": "center",
                              "text-halign": "center",
                              "text-outline-color": "#fff",
                              "text-outline-width": 2,
                              "font-size": 12,
                              width: 50,
                              height: 50,
                            },
                          },
                          {
                            selector: "edge",
                            style: {
                              "curve-style": "unbundled-bezier",
                              "control-point-step-size": 20,
                              "line-color": (edge) =>
                                edge.data("weight") > 0
                                  ? "rgba(144, 238, 144, 0.6)"
                                  : "rgba(255, 182, 193, 0.6)",
                              width: (edge) =>
                                Math.abs(edge.data("weight")) * 15,
                              "target-arrow-color": (edge) =>
                                edge.data("weight") > 0
                                  ? "rgba(144, 238, 144, 0.8)"
                                  : "rgba(255, 182, 193, 0.8)",
                              "target-arrow-shape": "triangle",
                              label: "data(label)",
                              "font-size": 10,
                              "text-background-color": "#fff",
                              "text-background-opacity": 0.8,
                            },
                          },
                        ]}
                      />
                    </div>

                    <div
                      style={{
                        flex: 1,
                        padding: "16px",
                        backgroundColor: "#f9f9f9",
                        border: "1px solid #ddd",
                        borderRadius: "8px",
                        marginLeft: "16px",
                        height: "250px",
                      }}
                    >
                      <Typography variant="h6" gutterBottom>
                        Legend
                      </Typography>
                      <div style={{ marginBottom: "8px" }}>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            marginBottom: "4px",
                          }}
                        >
                          <div
                            style={{
                              backgroundColor: "#FAFAD2",
                              width: "20px",
                              height: "20px",
                              borderRadius: "50%",
                              marginRight: "8px",
                              border: "1px solid black",
                            }}
                          ></div>
                          <Typography>Sensitive Attribute</Typography>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            marginBottom: "4px",
                          }}
                        >
                          <div
                            style={{
                              backgroundColor: "#87CEFA",
                              width: "20px",
                              height: "20px",
                              borderRadius: "50%",
                              marginRight: "8px",
                              border: "1px solid black",
                            }}
                          ></div>
                          <Typography>Target Attribute</Typography>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            marginBottom: "4px",
                          }}
                        >
                          <div
                            style={{
                              backgroundColor: "#EDEDED",
                              width: "20px",
                              height: "20px",
                              borderRadius: "50%",
                              marginRight: "8px",
                              border: "1px dashed black",
                            }}
                          ></div>
                          <Typography>Confounder</Typography>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            marginBottom: "4px",
                          }}
                        >
                          <div
                            style={{
                              backgroundColor: "#DCDCDC",
                              width: "20px",
                              height: "20px",
                              borderRadius: "50%",
                              marginRight: "8px",
                              border: "1px solid black",
                            }}
                          ></div>
                          <Typography>Default Node</Typography>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            marginBottom: "4px",
                          }}
                        >
                          <div
                            style={{
                              backgroundColor: "rgba(144, 238, 144, 0.6)",
                              width: "20px",
                              height: "4px",
                              marginRight: "8px",
                            }}
                          ></div>
                          <Typography>Positive Influence</Typography>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                          }}
                        >
                          <div
                            style={{
                              backgroundColor: "rgba(255, 182, 193, 0.6)",
                              width: "20px",
                              height: "4px",
                              marginRight: "8px",
                            }}
                          ></div>
                          <Typography>Negative Influence</Typography>
                        </div>
                      </div>
                    </div>
                  </div>
                </Paper>
              </motion.div>
            ) : (
              <Typography variant="body2" align="center"></Typography>
            )}
          </AnimatePresence>
          {/* Pop-Up Modal */}
          <Modal open={isMaximized} onClose={() => setIsMaximized(false)}>
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "90vw",
                height: "90vh",
                backgroundColor: "#fff",
                border: "1px solid #ddd",
                borderRadius: "8px",
                padding: "16px",
                boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.25)",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <Typography
                variant="h6"
                gutterBottom
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  mb: 2,
                }}
              >
                Causal Graph
                <IconButton onClick={() => setIsMaximized(false)}>
                  <FullscreenExitIcon /> {/* Minimize Icon */}
                </IconButton>
              </Typography>

              {/* Graph Container */}
              <Box sx={{ display: "flex", flex: 1, height: "100%" }}>
                {/* Enlarged Graph */}
                <Box sx={{ flex: 4, height: "100%" }}>
                  <CytoscapeComponent
                    elements={[
                      {
                        data: { id: sensitiveAttr, label: sensitiveAttr },
                        position: { x: 100, y: 250 },
                      },
                      {
                        data: { id: targetAttr, label: targetAttr },
                        position: { x: 900, y: 250 },
                      },
                      ...graphData.nodes
                        .filter(
                          (node) =>
                            node.id !== sensitiveAttr && node.id !== targetAttr
                        )
                        .map((node, index, array) => ({
                          data: {
                            id: node.id,
                            label: node.id,
                            isConfounder: confounders.includes(node.id),
                          },
                          position: {
                            x: 200 + (index * 700) / array.length,
                            y: 100 + (index % 5) * 80,
                          },
                        })),
                      ...graphData.links.map((link) => ({
                        data: {
                          source: link.source,
                          target: link.target,
                          weight: link.weight,
                          label: `Weight: ${link.weight.toFixed(2)}`,
                        },
                      })),
                    ]}
                    style={{
                      width: "100%",
                      height: "100%",
                    }}
                    layout={{
                      name: "preset",
                    }}
                    stylesheet={[
                      {
                        selector: "node",
                        style: {
                          "background-color": (node) =>
                            node.data("id") === sensitiveAttr
                              ? "#FAFAD2"
                              : node.data("id") === targetAttr
                              ? "#87CEFA"
                              : "#DCDCDC",
                          "border-width": 2,
                          "border-color": "black",
                          "border-style": (node) =>
                            node.data("isConfounder") ? "dashed" : "solid",
                          label: "data(label)",
                          "text-valign": "center",
                          "text-halign": "center",
                          "text-outline-color": "#fff",
                          "text-outline-width": 2,
                          "font-size": 16, // Bigger text in modal
                          width: 70, // Larger nodes
                          height: 70,
                        },
                      },
                      {
                        selector: "edge",
                        style: {
                          "curve-style": "unbundled-bezier",
                          "control-point-step-size": 30,
                          "line-color": (edge) =>
                            edge.data("weight") > 0
                              ? "rgba(144, 238, 144, 0.6)"
                              : "rgba(255, 182, 193, 0.6)",
                          width: (edge) => Math.abs(edge.data("weight")) * 15,
                          "target-arrow-color": (edge) =>
                            edge.data("weight") > 0
                              ? "rgba(144, 238, 144, 0.8)"
                              : "rgba(255, 182, 193, 0.8)",
                          "target-arrow-shape": "triangle",
                          label: "data(label)",
                          "font-size": 14, // Bigger edge labels
                          "text-background-color": "#fff",
                          "text-background-opacity": 0.8,
                        },
                      },
                    ]}
                  />
                </Box>

                {/* Enlarged Legend */}
                <Box
                  sx={{
                    flex: 1,
                    padding: "16px",
                    backgroundColor: "#f9f9f9",
                    border: "1px solid #ddd",
                    borderRadius: "8px",
                    marginLeft: "16px",
                  }}
                >
                  <Typography variant="h6" gutterBottom>
                    Legend
                  </Typography>
                  <Box sx={{ marginBottom: "8px" }}>
                    {[
                      { color: "#FAFAD2", label: "Sensitive Attribute" },
                      { color: "#87CEFA", label: "Target Attribute" },
                      { color: "#EDEDED", label: "Confounder", dashed: true },
                      { color: "#DCDCDC", label: "Default Node" },
                    ].map(({ color, label, dashed }) => (
                      <Box
                        key={label}
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          mb: 1,
                        }}
                      >
                        <Box
                          sx={{
                            backgroundColor: color,
                            width: "20px",
                            height: "20px",
                            borderRadius: "50%",
                            marginRight: "8px",
                            border: `1px ${dashed ? "dashed" : "solid"} black`,
                          }}
                        />
                        <Typography>{label}</Typography>
                      </Box>
                    ))}
                    {[
                      {
                        color: "rgba(144, 238, 144, 0.6)",
                        label: "Positive Influence",
                      },
                      {
                        color: "rgba(255, 182, 193, 0.6)",
                        label: "Negative Influence",
                      },
                    ].map(({ color, label }) => (
                      <Box
                        key={label}
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          mb: 1,
                        }}
                      >
                        <Box
                          sx={{
                            backgroundColor: color,
                            width: "20px",
                            height: "4px",
                            marginRight: "8px",
                          }}
                        />
                        <Typography>{label}</Typography>
                      </Box>
                    ))}
                  </Box>
                </Box>
              </Box>
            </Box>
          </Modal>

          {visGraph ? (
            <Paper
              elevation={3}
              sx={{
                p: 4,
                borderRadius: theme.shape.borderRadius,
                backgroundColor: theme.palette.background.paper,
                width: "400px",
                height: "500px",
                align: "start",
                alignSelf: "start",
                gridColumn: "1 / 2",
                gridRow: "4 / 5",
              }}
            >
              <Typography variant="h6" gutterBottom>
                Mediator Variables
              </Typography>

              <div style={{ height: "100%", width: "100%" }}>
                <CytoscapeComponent
                  elements={[
                    {
                      data: { id: sensitiveAttr, label: sensitiveAttr },
                    },
                    ...mediators.map((mediator) => ({
                      data: { id: mediator, label: mediator, isMediator: true },
                    })),
                    {
                      data: { id: targetAttr, label: targetAttr },
                    },
                    ...mediators.map((mediator) => ({
                      data: {
                        source: sensitiveAttr,
                        target: mediator,
                        label: "",
                      },
                    })),
                    ...mediators.map((mediator) => ({
                      data: {
                        source: mediator,
                        target: targetAttr,
                        label: "",
                      },
                    })),
                  ]}
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                  layout={{
                    name: "dagre",
                    rankDir: "LR",
                    nodeSep: 100,
                    edgeSep: 10,
                  }}
                  stylesheet={[
                    {
                      selector: "node",
                      style: {
                        "background-color": (node) =>
                          node.data("id") === sensitiveAttr
                            ? "#FAFAD2"
                            : node.data("id") === targetAttr
                            ? "#87CEFA"
                            : node.data("isMediator")
                            ? "#fad2f7"
                            : "#DCDCDC",
                        "border-width": 2,
                        "border-color": "black",
                        label: "data(label)",
                        "text-valign": "center",
                        "text-halign": "center",
                        "text-outline-color": "#fff",
                        "text-outline-width": 2,
                        "font-size": 12,
                        width: 50,
                        height: 50,
                        shape: "ellipse",
                      },
                    },
                    {
                      selector: "edge",
                      style: {
                        "curve-style": "straight",
                        "line-color": "#f0d343",
                        width: 3,
                        "target-arrow-color": "#f0d343",
                        "target-arrow-shape": "triangle",
                        label: "data(label)",
                        "font-size": 10,
                        "text-background-color": "#fff",
                        "text-background-opacity": 0.8,
                      },
                    },
                  ]}
                  cy={(cy) => {
                    cy.userZoomingEnabled(false);
                    cy.panningEnabled(false);
                    cy.boxSelectionEnabled(false);
                    cy.autolock(true);
                  }}
                />
              </div>
            </Paper>
          ) : (
            <Typography variant="body2" align="center"></Typography>
          )}
          {visGraph && (
            <Paper
              elevation={3}
              sx={{
                p: 4,
                borderRadius: theme.shape.borderRadius,
                backgroundColor: theme.palette.background.paper,
                // width: "1200px",
                align: "start",
                alignSelf: "start",
                gridColumn: "2 / 3",
                gridRow: "4 / 5",
              }}
            >
              <Typography variant="h5" gutterBottom>
                Causal Metrics Dashboard
              </Typography>
              <Grid2 container spacing={3}>
                {/* No Proxy Discrimination */}
                <Grid2 item xs={12}>
                  <Paper elevation={3} sx={{ p: 2 }}>
                    <Typography variant="h6">
                      No Proxy Discrimination
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      Ensure no paths pass through proxy variables.
                    </Typography>
                    {noProxyDiscrimination !== null && (
                      <Typography
                        sx={{
                          fontWeight: "bold",
                          color: noProxyDiscrimination
                            ? "rgba(144, 238, 144, 0.8)"
                            : "rgba(255, 99, 71, 0.8)",
                        }}
                      >
                        {noProxyDiscrimination
                          ? "No proxy discrimination detected."
                          : "Proxy discrimination exists."}
                      </Typography>
                    )}
                    {!noProxyDiscrimination &&
                      proxyDiscriminationPaths.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography
                            variant="body2"
                            sx={{ mb: 1, fontWeight: "bold" }}
                          >
                            Failing Paths:
                          </Typography>
                          {proxyDiscriminationPaths.map((pathObj, index) => (
                            <Typography
                              key={index}
                              variant="body2"
                              sx={{
                                padding: "4px",
                                borderRadius: "4px",
                                backgroundColor: "rgba(255, 182, 193, 0.2)",
                                mb: 1,
                              }}
                            >
                              <b>Path:</b> {pathObj.path.join(" → ")} <br />
                              {/* <b>Reason:</b> {pathObj.reason} */}
                            </Typography>
                          ))}
                        </Box>
                      )}
                    <Typography variant="body2" sx={{ mt: 2 }}>
                      Select Proxy Variables:
                    </Typography>
                    <Box
                      sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1 }}
                    >
                      {attributes
                        .filter(
                          (attr) =>
                            attr !== sensitiveAttr && attr !== targetAttr
                        )
                        .map((attr) => (
                          <Chip
                            key={attr}
                            label={attr}
                            color={
                              proxyVars.includes(attr) ? "primary" : "default"
                            }
                            onClick={() => {
                              const updatedVars = proxyVars.includes(attr)
                                ? proxyVars.filter((v) => v !== attr)
                                : [...proxyVars, attr];
                              setProxyVars(updatedVars);
                              handleNoProxyDiscrimination(updatedVars);
                            }}
                          />
                        ))}
                    </Box>
                  </Paper>
                </Grid2>

                {/* No Unresolved Discrimination */}
                <Grid2 item xs={12}>
                  <Paper elevation={3} sx={{ p: 2 }}>
                    <Typography variant="h6">
                      No Unresolved Discrimination
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      Ensure all paths pass through resolving variables.
                    </Typography>
                    {noUnresolvedDiscrimination !== null && (
                      <Typography
                        sx={{
                          fontWeight: "bold",
                          color: noUnresolvedDiscrimination
                            ? "rgba(144, 238, 144, 0.8)"
                            : "rgba(255, 99, 71, 0.8)",
                        }}
                      >
                        {noUnresolvedDiscrimination
                          ? "All paths resolved."
                          : "Some paths unresolved."}
                      </Typography>
                    )}
                    {!noUnresolvedDiscrimination &&
                      unresolvedDiscriminationPaths.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography
                            variant="body2"
                            sx={{ mb: 1, fontWeight: "bold" }}
                          >
                            Failing Paths:
                          </Typography>
                          {unresolvedDiscriminationPaths.map(
                            (pathObj, index) => (
                              <Typography
                                key={index}
                                variant="body2"
                                sx={{
                                  padding: "4px",
                                  borderRadius: "4px",
                                  backgroundColor: "rgba(255, 182, 193, 0.2)",
                                  mb: 1,
                                }}
                              >
                                <b>Path:</b> {pathObj.path.join(" → ")} <br />
                                {/* <b>Reason:</b> {pathObj.reason} */}
                              </Typography>
                            )
                          )}
                        </Box>
                      )}
                    <Typography variant="body2" sx={{ mt: 2 }}>
                      Select Resolving Variables:
                    </Typography>
                    <Box
                      sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1 }}
                    >
                      {attributes
                        .filter(
                          (attr) =>
                            attr !== sensitiveAttr && attr !== targetAttr
                        )
                        .map((attr) => (
                          <Chip
                            key={attr}
                            label={attr}
                            color={
                              resolvingVars.includes(attr)
                                ? "primary"
                                : "default"
                            }
                            onClick={() => {
                              const updatedVars = resolvingVars.includes(attr)
                                ? resolvingVars.filter((v) => v !== attr)
                                : [...resolvingVars, attr];
                              setResolvingVars(updatedVars);
                              handleNoUnresolvedDiscrimination(updatedVars);
                            }}
                          />
                        ))}
                    </Box>
                  </Paper>
                </Grid2>

                {/* Path-Specific Fairness */}
                {/* {noProxyDiscrimination !== null &&
          noUnresolvedDiscrimination !== null && (
            <Grid2 item xs={12}>
              <Paper elevation={3} sx={{ p: 2 }}>
                <Typography variant="h6">Path-Specific Fairness</Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  Assess direct and indirect discrimination paths, and
                  explainable bias.
                </Typography>
                {pathSpecificFairness && (
                  <Box sx={{ mt: 2 }}>
                    <Typography>
                      <b>Direct Paths:</b>{" "}
                      {pathSpecificFairness.direct.length}
                    </Typography>
                    <Typography>
                      <b>Indirect Paths:</b>{" "}
                      {pathSpecificFairness.indirect.length}
                    </Typography>
                    <Typography>
                      <b>Explainable Bias:</b>{" "}
                      {pathSpecificFairness.explainable.length}
                    </Typography>
                  </Box>
                )}
                <Typography variant="body2" sx={{ mt: 2 }}>
                  Select Legitimate Variables:
                </Typography>
                <Box
                  sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1 }}
                >
                  {attributes
                    .filter(
                      (attr) => attr !== sensitiveAttr && attr !== targetAttr
                    )
                    .map((attr) => (
                      <Chip
                        key={attr}
                        label={attr}
                        color={
                          legitimateVars.includes(attr) ? "primary" : "default"
                        }
                        onClick={() => {
                          const updatedVars = legitimateVars.includes(attr)
                            ? legitimateVars.filter((v) => v !== attr)
                            : [...legitimateVars, attr];
                          setLegitimateVars(updatedVars);
                          handlePathSpecificFairness(updatedVars);
                        }}
                      />
                    ))}
                </Box>
              </Paper>
            </Grid2>
          )} */}
              </Grid2>
            </Paper>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default DataAnalysis;
