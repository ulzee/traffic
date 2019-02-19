
### Models

#### Rolling (Spatial)

Recurrently unrolls over stops upstream. Advantage is stops can be forecasted arbitrarily far back. However, timesteps are not treated discretely but instead as a feature vector.

##### models/RNN.py
##### models/Conv.py

#### Fixed (Temporal)


Recurrently unrolls over time up to current time. Advantage is stops can be forecasted arbitrarily far back. However, timesteps are not treated discretely but instead as a feature vector.

##### models/Cast.py

#### Other

No particular structure and treats the prediction as a arbitrary transformation of input data.

##### models/Linear.py

### Datasets

##### Routes

Iterates over **all** segments in entire dataset.

##### LocalRoute

Iterates over segments found in a single route. Route specified by name (i.e. *Bx15_0*)

##### SingleStop

Iterates over segments belonging to a single stop over all of time. Route specified by name (i.e. *Bx15_0*) and segment specified by *n*th in order observed.