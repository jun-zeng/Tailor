## Training Data
We have prepared training data of the OJClone and BigCloneBench
datasets. Please download them with the following link:
https://drive.google.com/file/d/1sQuMFwuelufxoP3_iAbpYO2rfdc3OzPU/edit,
https://drive.google.com/file/d/1u9s4K43NluxFMVLKoqFoLVJIKRIANuA2/edit, https://drive.google.com/file/d/1AQmGqxsMavWbd0fkphHRNr-nXic9wcj8/edit.

## Introduction
Input encodings for machine learning models:
* `typetoken2id.txt`
    * AST type/token file
    * The first line is the number of AST types+tokens
    * Each line is a AST type/token with its one-hot encoding (`TypeID` and
      `AST_type`) or (`TokenID` and `AST_token`)

* `entity2id.txt`
    * CPG entity file
    * The first line is the number of CPG entities
    * Each line is a CPG entity with its one-hot encoding (`EntityID` and `CPG_entity`)

* `rel2id.txt`
    * CPG relation file
    * The first line is the number of CPG relations
    * Each line is a CPG relation with its one-hot encoding (`RelationID` and
      `AST`)
    * We hardcode AST, CFG, PDG as 0, 1, 2

* `entity2typetoken.txt`
    * CPG entity to AST type/token file
    * The first line is the number of CPG entities
    * Each line is a CPG entity with its AST type/token (`EntityID`, `TypeID`,
      `TokenID`)
    * If an entity does not have a TokenID, it is represented as (`EntityID`,
      `TypeID`, -1)

* `stat2entity.txt`
    * Program statement to CPG entities
    * The first line is the number of program statements
    * We represent each program statement with its root entity in a CPG
    * Each line is a program statement and its CPG entities (`EntityID`, `EntityID` ...)

* `triple2id.txt`
    * CPG edge file
    * The first line is the number of edges in CPG
    * Each line is a CPG edge (`EntityID`, `EntityID`, `RelationID`)

* `typetoken_seq.txt`
    * AST type/token sequence file
    * The first line is the number of program functions
    * Each line is a sequence of AST types/tokens for a program function (`TypeID`, `TokenID`...)