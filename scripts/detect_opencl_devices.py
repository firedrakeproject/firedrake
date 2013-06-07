try:
    import pyopencl as cl
    platforms = cl.get_platforms()
    ctxs = []
    for i, p in enumerate(platforms):
        for j in range(len(p.get_devices())):
            ctxs.append('%d:%d' % (i,j))
    print ' '.join(ctxs)
except ImportError:
    print ''
