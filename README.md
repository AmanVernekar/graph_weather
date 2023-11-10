# Graph Weather

Please also refer to original README [here](https://github.com/openclimatefix/graph_weather/blob/main/README.md).

### Description of selected files and folders (including those only present in idun:/local/scratch-2/asv34/graph_weather)

I've mostly included comments only for files that I've created/edited myself.

 - consts.txt - means, standard deviations and variances for all 42 variables across space and time of global training data (months 1, 4, 7, 10 in 2022) - use this to normalise global data while testing most of the models
 - uk_consts.txt - means, standard deviations and variances for all 42 variables across space and time of UK training data (months 1, 4, 7, 10 in 2022) - use this to normalise UK data while testing regional models
 - plot_graphs.py - used to plot loss vs epoch gprahs
 - test_models.py - used to test models; update means, variances and stdevs in this based on the respective values used to normalise training data for a given model
 - download_data.py - downloads data from Copernicus (one time step per download file)
	 - TODO: figure out if all time steps can be downloaded in one file and still be accessed easily for conversion into a numpy file for subsequent processing and training/testing
 - train_model.py - need to specify model_type (global/vector/diff_pool)
 - plots - directory of training plots (data and hyperparameters are in file names)
 - outputs - directory of output files during training (data and hyperparameters are in file names)
 - dataset - normalised .npy files for different months. In dataset/final, 'normed_w_uk_at_coarsen' indicates the coarsen value of the uk data used to normalise the global/europe data (when predicting uk values, uk data is used to normalise even global and europe data)
 - graph_weather
	 - data
		 - data_preprocessor.py
		 - dataloader.py
	 - models
		 - layers
			 - assimilator_decoder.py - code for "Decoder" section of the model
			 - assimilator_encoder.py - code for "Encoder" section of the model
			 - assimilator_region_decoder.py - version of assimilator_decoder.py for regional models
			 - decoder.py
			 - encoder.py
			 - graph_net_block.py - Builds the GNN + implements MLP and message passing; heavily edited from original to implement attention based on models at varying resolutions
			 - graph_pool.py - implements methods to generate a vector/pooled graph from a sparse model for attention with a dense model
			 - old_graph_net_block.py - legacy graph_net_block.py
			 - processor.py
			 - region_decoder.py - version of decoder.py for regional models
			 - region_encoder.py - version of encoder.py for regional models
		 - space_attention.py - implements MultiResoForecaster for multi-resolution models
		 - forecast.py - implements GraphWeatherForecaster (basic model - global, single step, single resolution)
		 - losses.py - (mostly) implements the NormalisedMSELoss function described in Keisler
			 - TODO: normalise based on changes in variables in a 3 hour window (read Keisler for more details)
		 - time_attention.py - implements ParallelForecaster for combining multiple time steps

### General Process

 1. Use download_data.py to download test/training data from Copernicus
 2. Use graph_weather/data/data_preprocessor.py to store all the data in one numpy file - makes training significantly faster than using multiple zarr files
	 - use the generated means and stdevs to normalise the data. store these to also normalise training data.
	 - also think about which data to use to normalise (ex - if using uk, global and europe data to predict uk data, it might make sense to normalise all of them using values generated for the uk data)
 3. Train using train_model.py:
	 -	choose regional or global
	 -	if using parallel forecaster, choose linear or params or simple_attention
	 -	if using multi reso forecaster, choose vector or diff_pool
	 -	(Combination of space and time attention hasn't been implemented yet)

### Overview of results so far
I couldn't complete proof of concept tests for space attention so only time attention has been rigourously tested. Using three timesteps in the model leads to a ~12% improvement over just one timestep (based on normalised MSE across all nodes and physical variables). As the number of parameters is increased, best performance is seen in the 'simple_attention' models.

### TODO

 1. Debug space_attention. Both the vector and graph pooling based methods run into many errors.
 2. Run proof of concept tests for both space attention methods. Pick models to compare against (Graphcast, 'local' Graphcast, original Keisler model, etc).
 3. Train on the complete dataset (~2TB).
 4. Current models only use 42 variables (6 variables at 7 pressure levels). Add land-sea mask and orography.
