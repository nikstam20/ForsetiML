import React, { useState } from "react";
import {
  Box,
  CssBaseline,
  Drawer,
  List,
  ListItem,
  ListItemText,
  Toolbar,
  Typography,
  Divider,
} from "@mui/material";
import DataAnalysis from "./DataAnalysis";
import ExperimentModelling from "./ExperimentModelling";
import Regression from "./Regression";
import History from "./History";
import FairnessGlossary from "./FairnessGlossary"; // Import the new component
import MenuBookIcon from "@mui/icons-material/MenuBook";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import ScienceIcon from "@mui/icons-material/Science";
import InfoIcon from "@mui/icons-material/Info"; // Icon for the glossary
import { useDataset } from "./DatasetContext";

const drawerWidth = 240;

function Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const { probType } = useDataset();

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: drawerWidth,
            boxSizing: "border-box",
            backgroundColor: "#fafafa",
            color: "#2D3748",
            borderRight: "1px solid rgba(0, 0, 0, 0.12)",
            paddingTop: "10px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
          },
        }}
      >
        <Box>
          <Toolbar sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography
              variant="h6"
              noWrap
              sx={{ fontSize: "1.25rem", fontWeight: 600 }}
            >
              ForsetiML
            </Typography>
            <img
              src="/logo.png"
              alt="ForsetiML Logo"
              style={{ width: "32px", height: "32px" }}
            />
          </Toolbar>

          <Divider sx={{ backgroundColor: "rgba(0, 0, 0, 0.12)" }} />

          <List sx={{ flexGrow: 1 }}>
            <ListItem
              button
              onClick={() => setActiveTab(0)}
              sx={{
                paddingLeft: 2,
                paddingRight: 2,
                backgroundColor: activeTab === 0 ? "#E3F2FD" : "transparent",
                color: activeTab === 0 ? "#1565C0" : "#2D3748",
                borderLeft: activeTab === 0 ? "4px solid #1565C0" : "none",
                transition: "all 0.3s ease-in-out",
                "&:hover": {
                  backgroundColor: "#BBDEFB",
                },
              }}
            >
              <TrendingUpIcon sx={{ mr: 2, fontSize: "1.25rem" }} />
              <ListItemText
                primary="Data Analysis"
                sx={{ fontSize: "1.1rem" }}
              />
            </ListItem>

            <Divider sx={{ backgroundColor: "rgba(0, 0, 0, 0.12)" }} />

            <ListItem
              button
              onClick={() => setActiveTab(1)}
              sx={{
                paddingLeft: 2,
                paddingRight: 2,
                backgroundColor: activeTab === 1 ? "#E3F2FD" : "transparent",
                color: activeTab === 1 ? "#1565C0" : "#2D3748",
                borderLeft: activeTab === 1 ? "4px solid #1565C0" : "none",
                transition: "all 0.3s ease-in-out",
                "&:hover": {
                  backgroundColor: "#BBDEFB",
                },
              }}
            >
              <ScienceIcon sx={{ mr: 2, fontSize: "1.25rem" }} />
              <ListItemText
                primary="Experiment Modelling"
                sx={{ fontSize: "1.1rem" }}
              />
            </ListItem>

            <Divider sx={{ backgroundColor: "rgba(0, 0, 0, 0.12)" }} />

            <ListItem
              button
              onClick={() => setActiveTab(2)}
              sx={{
                paddingLeft: 2,
                paddingRight: 2,
                backgroundColor: activeTab === 2 ? "#E3F2FD" : "transparent",
                color: activeTab === 2 ? "#1565C0" : "#2D3748",
                borderLeft: activeTab === 2 ? "4px solid #1565C0" : "none",
                transition: "all 0.3s ease-in-out",
                "&:hover": {
                  backgroundColor: "#BBDEFB",
                },
              }}
            >
              <MenuBookIcon sx={{ mr: 2, fontSize: "1.25rem" }} />
              <ListItemText primary="History" sx={{ fontSize: "1.1rem" }} />
            </ListItem>
          </List>

          <Divider sx={{ backgroundColor: "rgba(0, 0, 0, 0.12)" }} />
        </Box>

        {/* Fairness Glossary tab */}
        <Box sx={{ position: "absolute", bottom: 20, width: "100%" }}>
          <ListItem
            button
            onClick={() => setActiveTab(3)}
            sx={{
              paddingLeft: 2,
              paddingRight: 2,
              backgroundColor: activeTab === 3 ? "#E3F2FD" : "transparent",
              color: activeTab === 3 ? "#1565C0" : "#2D3748",
              borderLeft: activeTab === 3 ? "4px solid #1565C0" : "none",
              transition: "all 0.3s ease-in-out",
              "&:hover": {
                backgroundColor: "#BBDEFB",
              },
              fontSize: "0.9rem",
              paddingY: "8px",
            }}
          >
            <InfoIcon sx={{ mr: 1, fontSize: "1rem" }} />
            <ListItemText
              primary="Fairness Glossary"
              sx={{ fontSize: "0.95rem" }}
            />
          </ListItem>
        </Box>
      </Drawer>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          bgcolor: "background.default",
          p: 4,
          ml: `${drawerWidth}px`,
          transition: "all 0.3s ease",
          fontSize: "1.2rem",
        }}
      >
        <Box sx={{ display: activeTab === 0 ? "block" : "none" }}>
          <DataAnalysis />
        </Box>
        <Box sx={{ display: activeTab === 1 ? "block" : "none" }}>
          {probType === "regression" ? <Regression /> : <ExperimentModelling />}
        </Box>
        <Box sx={{ display: activeTab === 2 ? "block" : "none" }}>
          <History />
        </Box>
        <Box sx={{ display: activeTab === 3 ? "block" : "none" }}>
          <FairnessGlossary />
        </Box>
      </Box>
    </Box>
  );
}

export default Dashboard;
