package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.math.Point;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntFunction;

public abstract class ImgViewLayerBase extends LayerBase {
    private double negativeBias = 255;
    private boolean wrap;
    private int sizeX;
    private int sizeY;
    protected int[] channelSelector;

    public ImgViewLayerBase() {
        super();
    }

    public ImgViewLayerBase(@Nonnull JsonObject json) {
        super(json);
        setSizeX(json.getAsJsonPrimitive("sizeX").getAsInt());
        setSizeY(json.getAsJsonPrimitive("sizeY").getAsInt());
        setNegativeBias(json.getAsJsonPrimitive("negativeBias").getAsDouble());
        JsonArray _channelPermutationFilter = json.getAsJsonArray("channelPermutationFilter");
        if (null != _channelPermutationFilter) {
            int[] channelSelector1 = new int[_channelPermutationFilter.size()];
            setChannelSelector(channelSelector1);
            for (int i = 0; i < getChannelSelector().length; i++) {
                getChannelSelector()[i] = _channelPermutationFilter.get(i).getAsInt();
            }
        }
        JsonPrimitive toroidal = json.getAsJsonPrimitive("wrap");
        this.setWrap(null != toroidal && toroidal.getAsBoolean());
    }

    private static void set(@Nonnull Tensor tensor,
                            int width, int height,
                            int x, int y,
                            int channel, boolean wrap,
                            double value) {
        try {
            assert channel >= 0 : channel;
            if (wrap) {
                while (x < 0)
                    x += width;
                x %= width;
                while (y < 0)
                    y += height;
                y %= height;
            }
            if (x < 0) {
                return;
            } else if (x >= width) {
                return;
            }
            if (y < 0) {
                return;
            } else if (y >= height) {
                return;
            }
            tensor.set(x, y, channel, value);
        } finally {
            tensor.freeRef();
        }
    }


    @Nonnull
    @Override
    public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
        @Nonnull final JsonObject json = super.getJsonStub();
        json.addProperty("sizeX", getSizeX());
        json.addProperty("sizeY", getSizeY());
        json.addProperty("negativeBias", getNegativeBias());
        json.addProperty("wrap", isWrap());
        if (null != getChannelSelector()) {
            JsonArray _channelPermutationFilter = new JsonArray();
            for (int i : getChannelSelector()) {
                _channelPermutationFilter.add(i);
            }
            json.add("channelSelector", _channelPermutationFilter);
        }
        return json;
    }

    private static double get(@Nonnull @RefIgnore Tensor tensor, int width, int height, int x, int y, int channel, boolean wrap) {
        assert channel >= 0 : channel;
        if (wrap) {
            while (x < 0)
                x += width;
            x %= width;
            while (y < 0)
                y += height;
            y %= height;
        }
        if (x < 0) {
            return 0.0;
        } else if (x >= width) {
            return 0.0;
        }
        if (y < 0) {
            return 0.0;
        } else if (y >= height) {
            return 0.0;
        }
        return tensor.get(x, y, channel);
    }

    /**
     * Get channel selector int [ ].
     *
     * @return the int [ ]
     */
    public int[] getChannelSelector() {
        return channelSelector;
    }

    /**
     * Sets channel selector.
     *
     * @param channelSelector the channel selector
     */
    public void setChannelSelector(int[] channelSelector) {
        this.channelSelector = channelSelector;
    }

    /**
     * Gets negative bias.
     *
     * @return the negative bias
     */
    public double getNegativeBias() {
        return negativeBias;
    }

    /**
     * Sets negative bias.
     *
     * @param negativeBias the negative bias
     */
    public void setNegativeBias(double negativeBias) {
        this.negativeBias = negativeBias;
    }

    /**
     * Gets size x.
     *
     * @return the size x
     */
    public int getSizeX() {
        return sizeX;
    }

    /**
     * Sets size x.
     *
     * @param sizeX the size x
     */
    public void setSizeX(int sizeX) {
        this.sizeX = sizeX;
    }

    /**
     * Gets size y.
     *
     * @return the size y
     */
    public int getSizeY() {
        return sizeY;
    }

    /**
     * Sets size y.
     *
     * @param sizeY the size y
     */
    public void setSizeY(int sizeY) {
        this.sizeY = sizeY;
    }

    /**
     * Is wrap boolean.
     *
     * @return the boolean
     */
    public boolean isWrap() {
        return wrap;
    }

    /**
     * Sets wrap.
     *
     * @param wrap the wrap
     */
    public void setWrap(boolean wrap) {
        this.wrap = wrap;
    }

    @Nonnull
    @Override
    public Result eval(@Nonnull final Result... inObj) {
        final Result input = inObj[0].addRef();
        RefUtil.freeRef(inObj);
        final TensorList batch = input.getData();
        boolean alive = input.isAlive();
        Result.Accumulator accumulator = input.getAccumulator();
        input.freeRef();
        @Nonnull final int[] inputDims = batch.getDimensions();
        return new Result(fwd(batch), new Accumulator(addRef(), inputDims, accumulator, alive), alive);
    }

    @NotNull
    public TensorArray fwd(TensorList batch) {
        @Nonnull final int[] inputDims = batch.getDimensions();
        assert 3 == inputDims.length;
        final @Nonnull int[] dimOut = getViewDimensions(inputDims);
        return new TensorArray(RefIntStream.range(0, batch.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                    @Nonnull final Tensor outputData = new Tensor(dimOut);
                    fwd(batch.get(dataIndex), outputData.addRef());
                    return outputData;
                }, batch)).toArray(Tensor[]::new));
    }

    @Nonnull
    protected int[] getViewDimensions(@Nonnull int @NotNull [] inputDims) { return inputDims; }

    @Nonnull
    @Override
    public RefList<double[]> state() {
        return new RefArrayList<>();
    }

    public @SuppressWarnings("unused")
    void _free() {
        super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    ImgViewLayerBase addRef() {
        return (ImgViewLayerBase) super.addRef();
    }

    /**
     * Fwd.
     *
     * @param inputData  the input data
     * @param outputData the output data
     */
    protected void fwd(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
        int[] inputDims = inputData.getDimensions();
        @Nonnull final int[] inDim = inputDims;
        @Nonnull final int[] outDim = outputData.getDimensions();
        assert 3 == inDim.length;
        assert 3 == outDim.length;
        assert inDim[2] == outDim[2] : RefArrays.toString(inDim) + "; " + RefArrays.toString(outDim);
        outputData.coordStream(true).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
            int[] coords = c.getCoords();
            Point xy = coordinateMapping(new Point(coords[0], coords[1]));
            int x = (int) Math.floor(xy.x);
            int y = (int) Math.floor(xy.y);
            int channel;
            if (null != channelSelector)
                channel = channelSelector[coords[2]];
            else
                channel = coords[2] + 1;
            final double value;
            if (0 < channel) {
                value = ImgViewLayerBase.get(inputData, inputDims[0], inputDims[1], x, y, channel - 1, wrap);
            } else {
                value = getNegativeBias() -
                        ImgViewLayerBase.get(inputData, inputDims[0], inputDims[1], x, y, -channel - 1, wrap);
            }
            outputData.set(c, value);
        }, inputData, outputData));
    }

    /**
     * Bck.
     *
     * @param outputDelta the output delta
     * @param inputDelta  the input delta
     */
    protected void bck(@Nonnull final Tensor outputDelta, @Nonnull final Tensor inputDelta) {
        int[] outDeltaDims = outputDelta.getDimensions();
        @Nonnull final int[] inputDeltaDims = inputDelta.getDimensions();
        assert 3 == outDeltaDims.length;
        assert 3 == inputDeltaDims.length;
        assert outDeltaDims[2] == inputDeltaDims[2] : RefArrays.toString(outDeltaDims) + "; "
                + RefArrays.toString(inputDeltaDims);
        outputDelta.coordStream(true).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
            double value = outputDelta.get(c);
            if (value == 0.0) return;
            int[] outCoord = c.getCoords();
            Point inCoords = coordinateMapping(new Point(outCoord[0], outCoord[1]));
            int x = (int) Math.floor(inCoords.x);
            int y = (int) Math.floor(inCoords.y);
            int channel;
            if (null != channelSelector)
                channel = channelSelector[outCoord[2]];
            else
                channel = outCoord[2] + 1;
            if (0 < channel) {
                ImgViewLayerBase.set(inputDelta.addRef(),
                        inputDeltaDims[0], inputDeltaDims[1],
                        x, y,
                        channel - 1,
                        wrap, value);
            } else {
                ImgViewLayerBase.set(
                        inputDelta.addRef(),
                        inputDeltaDims[0], inputDeltaDims[1],
                        x, y,
                        -channel - 1,
                        wrap, -value);
            }
        }, inputDelta, outputDelta));
    }

    protected abstract Point coordinateMapping(@Nonnull Point xy);

    private static class Accumulator extends Result.Accumulator {

        private final int[] inputDims;
        private ImgViewLayerBase imgViewLayer;
        private Result.Accumulator accumulator;
        private boolean alive;

        /**
         * Instantiates a new Accumulator.
         *
         * @param imgViewLayer the img view layer
         * @param inputDims    the input dims
         * @param accumulator  the accumulator
         * @param alive        the alive
         */
        public Accumulator(ImgViewLayerBase imgViewLayer, int[] inputDims, Result.Accumulator accumulator, boolean alive) {
            this.inputDims = inputDims;
            this.imgViewLayer = imgViewLayer;
            this.accumulator = accumulator;
            this.alive = alive;
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
            if (alive) {
                @Nonnull
                TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length())
                        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                            @Nullable final Tensor err = error.get(dataIndex);
                            @Nonnull final Tensor passback = new Tensor(inputDims);
                            imgViewLayer.bck(err.addRef(),
                                    passback.addRef());
                            err.freeRef();
                            return passback;
                        }, error.addRef())).toArray(Tensor[]::new));
                this.accumulator.accept(buffer == null ? null : buffer.addRef(), tensorArray);
            }
            error.freeRef();
            if (null != buffer)
                buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
            super._free();
            accumulator.freeRef();
            imgViewLayer.freeRef();
        }
    }
}
