OpenSWR-Mesa
============

Note: All development has moved into the Mesa repo.  This source is for reference only.
--------

Overview
--------

This is a repository of the integration work combining the high
performance, highly scalable core SWR rasterizer with Mesa.  A more
complete introduction and discussion towards upstreaming to the Mesa
project can be found on the mesa-dev mailing list.

Source is now upstreamed to the Mesa repo.  For the lastest source, please pull from:
* git://anongit.freedesktop.org/git/mesa/mesa

Notes
-----

* llvmpipe is the default software renderer.  Use
`GALLIUM_DRIVER=swr` to switch to the OpenSWR rasterizer.

* LLVM 3.6 or newer is required.

* To build SWR with autoconf, include the following in the config
line: `--with-gallium-drivers=swr --disable-dri --disable-egl --enable-xlib-glx`.

* SCons build support has been temporarily removed to focus on upstreaming.  SCons
will be reenabled at a later date.

* Code for the driver is in `src/gallium/drivers/swr`

* Code for the rasterizer is in `src/gallium/drivers/swr/rasterizer`
