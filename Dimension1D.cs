﻿namespace ArrayViewSketches
{
    public interface IStride1D
    {
        DimTuple1 Stride { get; }
    }

    public readonly struct GeneralStride1D : IStride1D
    {
        public GeneralStride1D(DimTuple1 stride)
        {
            Stride = stride;
        }

        public DimTuple1 Stride { get; }
    }

    public readonly struct DenseStride1D : IStride1D
    {
        public DimTuple1 Stride => 1;
    }

    public readonly struct ArrayView1D<TElem, TDim>
        where TElem : unmanaged
        where TDim : IStride1D
    {
        internal readonly ILGPU.ArrayView<TElem> Data;
        public readonly TDim Dim;

        public DimTuple1 Extent { get; }
        public DimTuple1 Stride => Dim.Stride;

        internal ArrayView1D(ILGPU.ArrayView<TElem> data, DimTuple1 extent, TDim dim)
        {
            Data = data;
            Extent = extent;
            Dim = dim;
        }

        public ArrayView1D<TElem, DenseStride1D> AsDense()
        {
            if (Dim.Stride.X != 1)
            {
                // throw new ApplicationException("ArrayView1D is not dense");
            }
            
            return new ArrayView1D<TElem, DenseStride1D>(Data, Extent, new DenseStride1D());
        }

        public ArrayView1D<TElem, TDim> SubView(DimTuple1 offset, DimTuple1 extent)
        {
            if (offset.X < 0 || offset.X + extent.X >= Extent.X)
            {
                // throw new IndexOutOfRangeException();
            }
            
            var linearIdx = offset.X * Stride.X;
            var data = Data.GetSubView(linearIdx);
            
            return new ArrayView1D<TElem, TDim>(data, extent, Dim);
        }
        
        public ref TElem this[DimTuple1 idx]
        {
            get
            {
                if (idx.X < 0 || idx.X > Extent.X)
                {
                    // throw new IndexOutOfRangeException();
                }

                var linearIdx = idx.X * Stride.X;
                return ref Data[linearIdx];
            }
        }
    }
}