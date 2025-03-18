import React, { createContext, useContext, useState } from "react";

const DatasetContext = createContext();

export const useDataset = () => useContext(DatasetContext);

export const DatasetProvider = ({ children }) => {
  const [dataset, setDataset] = useState(null);
  const [attributes, setAttributes] = useState([]);
  const [sensitiveAttr, setSensitiveAttr] = useState("");
  const [targetAttr, setTargetAttr] = useState("");
  const [privilegedGroup, setPrivilegedGroup] = useState(null);
  const [unprivilegedGroup, setUnprivilegedGroup] = useState(null);

  const [metrics, setMetrics] = useState(null);
  const [resultMetrics, setResultMetrics] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [results, setResults] = useState(null);
  const [baseTime, setBaseTime] = useState(null);
  const [baseSize, setBaseSize] = useState(null);
  const [methodType, setMethodType] = useState(null);
  const [queryResults, setQueryResults] = useState(null);
  const [userOptions, setUserOptions] = useState(null);
  const saveExperiment = (experimentDetails) => {
    setExperiments((prev) => [...prev, experimentDetails]);
  };
  const [limitations, setLimitations] = useState([]);
  const [probType, setProbType] = useState(null);

  return (
    <DatasetContext.Provider
      value={{
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
        metrics,
        setMetrics,
        resultMetrics,
        setResultMetrics,
        chartData,
        setChartData,
        experiments,
        saveExperiment,
        results,
        setResults,
        baseTime,
        setBaseTime,
        baseSize,
        setBaseSize,
        methodType,
        setMethodType,
        queryResults,
        setQueryResults,
        userOptions,
        setUserOptions,
        limitations,
        setLimitations,
        probType,
        setProbType,
      }}
    >
      {children}
    </DatasetContext.Provider>
  );
};
