using System;
using Accelerator = ILGPU.Runtime.Accelerator;

namespace ArrayViewSketches
{
    public class MemoryBuffer<TElem, TView> : IDisposable
        where TElem: unmanaged
        where TView: struct // Really want to constrain this to be some sort of ArrayView with the same TElem...
    {
        public MemoryBuffer(ILGPU.Runtime.MemoryBuffer<TElem> storage, TView rootView)
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
        public static MemoryBuffer<T, ArrayView1D<T, DenseDim1D>> Allocate1D<T>(this Accelerator a, DimTuple1 extent) where T : unmanaged
        {
            var storage = a.Allocate<T>(extent.X);
            var rootView = new ArrayView1D<T, DenseDim1D>(storage, extent, new DenseDim1D());
            return new MemoryBuffer<T, ArrayView1D<T, DenseDim1D>>(storage, rootView);
        }

        public static MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>> Allocate2D<T>(this Accelerator a, DimTuple2 extent) where T : unmanaged
        {
            return a.Allocate2DDenseX<T>(extent);
        }
        
        public static MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>> Allocate2DDenseX<T>(this Accelerator a, DimTuple2 extent) where T : unmanaged
        {
            var storage = a.Allocate<T>(extent.X * extent.Y);
            var rootView = new ArrayView2D<T, Stride2DDenseX>(storage, extent, new Stride2DDenseX(extent.X));
            return new MemoryBuffer<T, ArrayView2D<T, Stride2DDenseX>>(storage, rootView);
        }
        
        public static MemoryBuffer<T, ArrayView2D<T, Stride2DDenseY>> Allocate2DDenseY<T>(this Accelerator a, DimTuple2 extent) where T : unmanaged
        {
            var storage = a.Allocate<T>(extent.X * extent.Y);
            var rootView = new ArrayView2D<T, Stride2DDenseY>(storage, extent, new Stride2DDenseY(extent.X));
            return new MemoryBuffer<T, ArrayView2D<T, Stride2DDenseY>>(storage, rootView);
        }
    }
}