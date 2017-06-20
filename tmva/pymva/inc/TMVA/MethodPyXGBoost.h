// @(#)root/tmva/pymva $Id$
// Author: Enhao Song

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyXGBoost                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Interface for XGBoost python package.                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Enhao Song <es4re@virginia.edu> - UVA, U.S.A                              *
 *                                                                                *
 * Copyright (c) 2017:                                                            *
 *      CERN, Switzerland                                                         *
 *      Fermi Lab, U.S.A                                                          *
 *      UVA, U.S.A                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPyXGBoost
#define ROOT_TMVA_MethodPyXGBoost

#include "TMVA/PyMethodBase.h"

namespace TMVA {

   class MethodPyXGBoost : public PyMethodBase {

   public :

      // constructors
      MethodPyXGBoost(const TString &jobName,
            const TString &methodTitle,
            DataSetInfo &dsi,
            const TString &theOption = "");
      MethodPyXGBoost(DataSetInfo &dsi,
            const TString &theWeightFile);
      ~MethodPyXGBoost();

      void Train();
      void Init();
      void DeclareOptions();
      void ProcessOptions();

      // Check whether the given analysis type (regression, classification, ...)
      // is supported by this method
      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t);
      // Get signal probability of given event
      Double_t GetMvaValue(Double_t *errLower, Double_t *errUpper);
      std::vector<Double_t> GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t logProgress);
      // Get regression values of given event
      std::vector<Float_t>& GetRegressionValues();
      // Get class probabilities of given event
      std::vector<Float_t>& GetMulticlassValues();

      const Ranking *CreateRanking() { return 0; }
      virtual void TestClassification();
      virtual void AddWeightsXMLTo(void*) const{}
      virtual void ReadWeightsFromXML(void*){}
      virtual void ReadWeightsFromStream(std::istream&) {} // backward compatibility
      virtual void ReadWeightsFromStream(TFile&){} // backward compatibility
      void ReadModelFromFile();

      void GetHelpMessage() const;

    private:

      TString fFilenameModel; // Filename of the previously exported XGBoost model
      UInt_t fNumBoostRound {0}; // Number of boosting iterations
      Int_t fVerbose; // XGBoost verbosity during training
      Bool_t fContinueTraining; // Load weights from previous training
      Bool_t fSaveBestOnly; // Store only weights with smallest validation loss
      Int_t fTriesEarlyStopping; // Stop training if validation loss is not decreasing for several epochs
      TString fLearningRateSchedule; // Set new learning rate at specific epochs

      bool fModelIsSetup = false; // flag whether model is loaded, neede for getMvaValue during evaluation
      float* fVals = nullptr; // variables array used for GetMvaValue
      std::vector<float> fOutput; // probability or regression output array used for GetMvaValue
      UInt_t fNVars {0}; // number of variables
      UInt_t fNOutputs {0}; // number of outputs (classes or targets)
      TString fFilenameTrainedModel; // output filename for trained model

      void SetupXGBoostModel(Bool_t loadTrainedModel); // setups the needed variables loads the model

      ClassDef(MethodPyXGBoost, 0);
   };

} // namespace TMVA

#endif // ROOT_TMVA_MethodPyXGBoost
