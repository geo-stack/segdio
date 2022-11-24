<br />
<div align="center">
  <a href="https://www.geostack.ca/">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">SEGD IO</h3>
</div>

## About The Project
Reader for seismic tape files conforming to the SEG-D specifications.

### Currently supported
 - [SEG-D 2.1](https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_d_rev2.1.pdf) (24 and 32 bits)

## Usage

### Print file summary
```python
from read_segd import SEGD
segd = SEGD("segd_file_name.segd")
print(segd)
 ```

### Read trace data
This will return the data from the specified channel set.
```python
data = segd.data(CHANNEL_SET_INDEX)
 ```

### Read trace header
This will return the trace headers from the specified channel set.
```python
channel_trace_headers = segd.channel_set_headers[CHANNEL_SET_INDEX].trace_headers
 ```
 
## Acknowledgments
Adapted from https://github.com/drsudow/SEG-D (Author: [Mattias Südow](https://github.com/drsudow))