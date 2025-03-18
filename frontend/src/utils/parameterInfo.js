const parameterInfo = {
    booster: "Type of booster to use: 'gbtree', 'gblinear', or 'dart'.",
    eta: "Learning rate (default: 0.3). Lower values make learning slower but improve performance.",
    max_depth: "Maximum depth of the tree (default: 6). Deeper trees increase complexity.",
    min_child_weight: "Minimum sum of instance weight (hessian) in a child (default: 1).",
    gamma: "Minimum loss reduction required to split a node (default: 0).",
    subsample: "Fraction of samples used for training (default: 1.0).",
  };
  
  export default parameterInfo;
  