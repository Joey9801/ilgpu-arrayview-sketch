namespace ArrayViewSketches
{
    public interface IStride2D
    {
        DimTuple2 Stride { get;  }
    }

    public readonly struct Stride2DGeneral : IStride2D
    {
        public Stride2DGeneral(DimTuple2 stride)
        {
            Stride = stride;
        }
        
        public DimTuple2 Stride { get; }
    }

    public readonly struct Stride2DDenseX : IStride2D
    {
        private readonly long _yStride;
        
        public Stride2DDenseX(long yStride)
        {
            _yStride = yStride;
        }

        public DimTuple2 Stride => (1, _yStride);
    }

    public readonly struct Stride2DDenseY : IStride2D
    {
        private readonly long _xStride;
        
        public Stride2DDenseY(long xStride)
        {
            _xStride = xStride;
        }

        public DimTuple2 Stride => (_xStride, 1);
    }

    public readonly struct ArrayView2D<TElem, TDim>
        where TElem : unmanaged
        where TDim : IStride2D
    {
        internal readonly ILGPU.ArrayView<TElem> Data;
        public readonly TDim Dim;

        public DimTuple2 Extent { get; }
        public DimTuple2 Stride => Dim.Stride;
        
        internal ArrayView2D(ILGPU.ArrayView<TElem> data, DimTuple2 extent, TDim dim)
        {
            Data = data;
            Extent = extent;
            Dim = dim;
        }

        public ArrayView2D<TElem, Stride2DDenseX> AsDenseX()
        {
            if (Stride.X != 1)
            {
                // throw new ApplicationException("ArrayView2D is not dense in the X dimension");
            }
            
            return new ArrayView2D<TElem, Stride2DDenseX>(Data, Extent, new Stride2DDenseX(Stride.Y));
        }

        public ArrayView2D<TElem, Stride2DDenseY> AsDenseY()
        {
            if (Stride.Y != 1)
            {
                // throw new ApplicationException("ArrayView2D is not dense in the Y dimension");
            }
            
            return new ArrayView2D<TElem, Stride2DDenseY>(Data, Extent, new Stride2DDenseY(Stride.X));
        }

        public ArrayView2D<TElem, Stride2DGeneral> AsGeneral()
        {
            return new ArrayView2D<TElem, Stride2DGeneral>(Data, Extent, new Stride2DGeneral(Stride));
        }

        public ArrayView2D<TElem, TDim> SubView(DimTuple2 offset, DimTuple2 extent)
        {
            if (offset.X < 0 || offset.X + extent.X >= Extent.X ||
                offset.Y < 0 || offset.Y + extent.Y >= Extent.Y)
            {
                // throw new IndexOutOfRangeException();
            }

            var linearIdx = offset.X * Stride.X + offset.Y * Stride.Y;
            var data = Data.GetSubView(linearIdx);
            return new ArrayView2D<TElem, TDim>(data, extent, Dim);
        }

        public ref TElem this[DimTuple2 idx]
        {
            get
            {
                if (idx.X < 0 || idx.X > Extent.X ||
                    idx.Y < 0 || idx.Y > Extent.Y)
                {
                    // throw new IndexOutOfRangeException();
                }

                unsafe
                {
                    var linearIdx = idx.X * Stride.X + idx.Y * Stride.Y;
                    return ref Data[linearIdx];
                }
            }
        }
    }

    public static class ArrayView2DExtensions
    {
        // Take a 1D slice of a 2D array where we don't know the stride at compile time
        public static ArrayView1D<TElem, GeneralStride1D> SliceX<TElem, TDim>(
            this ArrayView2D<TElem, TDim> arr,
            long y)
        where TElem : unmanaged where TDim : IStride2D
        {
            if (y < 0 || y >= arr.Extent.Y)
            {
                // throw new IndexOutOfRangeException();
            }

            ILGPU.Interop.WriteLine("General purpose SliceX");

            var offset = y * arr.Stride.Y;
            var data = arr.Data.GetSubView(offset);
            var dim = new GeneralStride1D(arr.Stride.X);
            return new ArrayView1D<TElem, GeneralStride1D>(data, arr.Extent.X, dim);
        }
        
        // Take a dense 1D slice out of a 2D array that is known to be X-major
        public static ArrayView1D<TElem, DenseDim1D> SliceX<TElem>(
            this ArrayView2D<TElem, Stride2DDenseX> arr,
            long y)
        where TElem : unmanaged
        {
            if (y < 0 || y >= arr.Extent.Y)
            {
                // throw new IndexOutOfRangeException();
            }
            
            ILGPU.Interop.WriteLine("Specialized SliceX");

            var offset = y * arr.Stride.Y;
            var data = arr.Data.GetSubView(offset);
            var dim = new DenseDim1D();
            return new ArrayView1D<TElem, DenseDim1D>(data, arr.Extent.X, dim);
        }
        
        public static ArrayView1D<TElem, GeneralStride1D> SliceY<TElem, TDim>(
            this ArrayView2D<TElem, TDim> arr,
            long x)
        where TElem : unmanaged where TDim : IStride2D
        {
            if (x < 0 || x >= arr.Extent.X)
            {
                // throw new IndexOutOfRangeException();
            }

            var offset = x * arr.Stride.X;
            var data = arr.Data.GetSubView(offset);
            var dim = new GeneralStride1D(arr.Stride.Y);
            return new ArrayView1D<TElem, GeneralStride1D>(data, arr.Extent.Y, dim);
        }
        
        public static ArrayView1D<TElem, DenseDim1D> SliceY<TElem>(
            this ArrayView2D<TElem, Stride2DDenseY> arr,
            long x)
        where TElem : unmanaged
        {
            if (x < 0 || x >= arr.Extent.X)
            {
                // throw new IndexOutOfRangeException();
            }

            var offset = x * arr.Stride.X;
            var data = arr.Data.GetSubView(offset);
            var dim = new DenseDim1D();
            return new ArrayView1D<TElem, DenseDim1D>(data, arr.Extent.Y, dim);
        }
    }
}