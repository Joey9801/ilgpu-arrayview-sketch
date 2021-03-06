﻿using System;
using Accelerator = ILGPU.Runtime.Accelerator;

namespace ArrayViewSketches
{
    public class MemoryBuffer<TElem, TView> : IDisposable
        where TElem: unmanaged
        where TView: struct // Really want to constrain this to be some sort of ArrayView with the same TElem...
    {
        internal MemoryBuffer(ILGPU.Runtime.MemoryBuffer<TElem> storage, TView rootView)
        {
            _storage = storage;
            RootView = rootView;
        }

        private ILGPU.Runtime.MemoryBuffer<TElem> _storage;
        public readonly TView RootView;
        
        public void Dispose()
        {
            _storage?.Dispose();
            _storage = null;
        }
    }

    public static class Allocator
    {
        public static MemoryBuffer<T, ArrayView1D<T, Stride1DDense>> Allocate1D<T>(
            this Accelerator a,
            DimTuple1 extent)
        where T : unmanaged
        {
            var storage = a.Allocate<T>(extent.X);
            var rootView = new ArrayView1D<T, Stride1DDense>(storage, extent, new Stride1DDense());
            return new MemoryBuffer<T, ArrayView1D<T, Stride1DDense>>(storage, rootView);
        }

        public static MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>> Allocate2D<T>(
            this Accelerator a,
            DimTuple2 extent) 
        where T : unmanaged
        {
            return a.Allocate2DDenseX<T>(extent);
        }
        
        public static MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>> Allocate2DDenseX<T>(
            this Accelerator a,
            DimTuple2 extent)
        where T : unmanaged
        {
            var storage = a.Allocate<T>(extent.X * extent.Y);
            var rootView = new ArrayView2D<T, Stride2DDenseX>(storage, extent, new Stride2DDenseX(extent.X));
            return new MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>>(storage, rootView);
        }
        
        public static MemoryBuffer<T, ArrayView2D<T, Stride2DDenseY>> Allocate2DDenseY<T>(
            this Accelerator a,
            DimTuple2 extent)
        where T : unmanaged
        {
            var storage = a.Allocate<T>(extent.X * extent.Y);
            var rootView = new ArrayView2D<T, Stride2DDenseY>(storage, extent, new Stride2DDenseY(extent.X));
            return new MemoryBuffer<T, ArrayView2D<T, Stride2DDenseY>>(storage, rootView);
        }

        public static MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>> Allocate2DPitchedX<T>(
            this Accelerator a,
            DimTuple2 extent)
        where T : unmanaged
        {
            // Limitation: Since the underlying storage for our custom ArrayViews is well-typed, can only add padding
            // in increments of sizeof(T) bytes. This limitation could be lifted within this sketch with some unsafe
            // code that always used a backing store of type <byte> and casted pointers on every access.

            long unpitchedBytes;
            unsafe
            {
                if (sizeof(T) > 128 || 128 % sizeof(T) != 0)
                {
                    throw new ArgumentException($"Cannot perform pitched allocation for {nameof(T)}");
                }

                unpitchedBytes = extent.X * sizeof(T);
            }

            var pitchedBytes = ((unpitchedBytes - 1) / 128 + 1) * 128;
            var pitchedExtentX = pitchedBytes / 128;

            var storage = a.Allocate<T>(pitchedExtentX * extent.Y);
            var rootView = new ArrayView2D<T, Stride2DDenseX>(storage, extent, new Stride2DDenseX(pitchedExtentX));
            return new MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>>(storage, rootView);
        }
    }
}