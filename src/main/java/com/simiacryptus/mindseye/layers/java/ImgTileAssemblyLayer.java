/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.IntFunction;

/**
 * The type Img tile assembly layer.
 */
@SuppressWarnings("serial")
public class ImgTileAssemblyLayer extends LayerBase {

  private final int columns;
  private final int rows;
  private int paddingX = 0;
  private int paddingY = 0;
  private int offsetX = 0;
  private int offsetY = 0;

  /**
   * Instantiates a new Img tile assembly layer.
   *
   * @param columns the columns
   * @param rows    the rows
   */
  public ImgTileAssemblyLayer(final int columns, final int rows) {
    super();
    this.columns = columns;
    this.rows = rows;
  }

  /**
   * Instantiates a new Img tile assembly layer.
   *
   * @param json the json
   */
  protected ImgTileAssemblyLayer(@Nonnull final JsonObject json) {
    super(json);
    columns = getInt(json, "columns", 1);
    rows = getInt(json, "rows", 1);
    setPaddingX(getInt(json, "paddingX", 0));
    setPaddingY(getInt(json, "paddingY", 0));
    setOffsetX(getInt(json, "offsetX", 0));
    setOffsetY(getInt(json, "offsetY", 0));
  }

  /**
   * Gets offset x.
   *
   * @return the offset x
   */
  public int getOffsetX() {
    return offsetX;
  }

  /**
   * Sets offset x.
   *
   * @param offsetX the offset x
   */
  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
  }

  /**
   * Gets offset y.
   *
   * @return the offset y
   */
  public int getOffsetY() {
    return offsetY;
  }

  /**
   * Sets offset y.
   *
   * @param offsetY the offset y
   */
  public void setOffsetY(int offsetY) {
    this.offsetY = offsetY;
  }

  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public int getPaddingX() {
    return paddingX;
  }

  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   */
  public void setPaddingX(int paddingX) {
    this.paddingX = paddingX;
  }

  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public int getPaddingY() {
    return paddingY;
  }

  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   */
  public void setPaddingY(int paddingY) {
    this.paddingY = paddingY;
  }

  /**
   * Copy.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @param offsetX    the offset x
   * @param offsetY    the offset y
   * @param paddingX   the padding x
   * @param paddingY   the padding y
   * @param toroidal   the toroidal
   * @param rowF       the row f
   * @param colF       the col f
   */
  public static void copy(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData, final int offsetX,
                          final int offsetY, final int paddingX, final int paddingY, final boolean toroidal, final double rowF,
                          final double colF) {
    int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : RefArrays.toString(inDim) + "; " + RefArrays.toString(outDim);
    //    outputData.coordStream(true).forEach((outputCoord) -> {
    //      double value = getMaxValue(inputData, outputCoord, offsetX, offsetY, paddingX, paddingY, toroidal);
    //      if (Double.isFinite(value)) outputData.set(outputCoord, value);
    //    });
    inputData.coordStream(true).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) inputCoord -> {
      double inputValue = inputData.get(inputCoord);
      int inputWidth = inDim[0];
      int inputHeight = inDim[1];
      int outputWidth = outDim[0];
      int outputHeight = outDim[1];
      int x = inputCoord.getCoords()[0];
      int y = inputCoord.getCoords()[1];
      //      x += offsetX;
      //      y += offsetY;
      if (x < paddingX / 2 && colF > 0.0) {
        return;
      }
      if (y < paddingY / 2 && rowF > 0.0) {
        return;
      }
      if (x >= inputWidth - paddingX / 2 && colF < 1.0) {
        return;
      }
      if (y >= inputHeight - paddingY / 2 && rowF < 1.0) {
        return;
      }
      x += offsetX;
      y += offsetY;
      int z = inputCoord.getCoords()[2];
      if (toroidal) {
        while (x < 0)
          x += outputWidth;
        x %= outputWidth;
        while (y < 0)
          y += outputHeight;
        y %= outputHeight;
      }
      if (x < 0) {
        return;
      }
      if (y < 0) {
        return;
      }
      if (x >= outputWidth) {
        return;
      }
      if (y >= outputHeight) {
        return;
      }
      outputData.set(x, y, z, inputValue);
    }, outputData, inputData));
  }

  /**
   * From json img tile assembly layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img tile assembly layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgTileAssemblyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileAssemblyLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList in0Data = inObj[0].getData();
    boolean in0Alive = inObj[0].isAlive();
    assert 3 == in0Data.getDimensions().length;
    int length = in0Data.length();
    in0Data.freeRef();
    int[] outputDims = getOutputDims(RefUtil.addRef(inObj));
    TensorArray data = fwd(length, outputDims, RefUtil.addRef(inObj));
    Result.Accumulator accumulator = new Accumulator(columns, offsetX, offsetY, rows, ImgTileAssemblyLayer.this.getPaddingX(), ImgTileAssemblyLayer.this.getPaddingY(), inObj);
    return new Result(data, accumulator, in0Alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("columns", columns);
    json.addProperty("rows", rows);
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("offsetX", getOffsetX());
    json.addProperty("offsetY", getOffsetY());
    return json;
  }

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
  ImgTileAssemblyLayer addRef() {
    return (ImgTileAssemblyLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(int length, int[] outputDims, @Nonnull Result[] inObj) {
    return new TensorArray(RefIntStream.range(0, length).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nonnull final Tensor outputData = new Tensor(outputDims);

          int totalWidth = 0;
          int positionY = offsetY;
          int inputIndex = 0;
          for (int row = 0; row < rows; row++) {
            int positionX = offsetX;
            int rowHeight = 0;
            for (int col = 0; col < columns; col++) {
              TensorList tileTensor = inObj[inputIndex].getData();
              int[] tileDimensions = tileTensor.getDimensions();
              rowHeight = Math.max(rowHeight, tileDimensions[1]);
              Tensor inputData = tileTensor.get(dataIndex);
              tileTensor.freeRef();
              ImgTileAssemblyLayer.copy(inputData,
                  outputData.addRef(), positionX, positionY,
                  0 >= positionX ? 0 : getPaddingX() / 2, 0 >= positionY ? 0 : getPaddingY() / 2,
                  offsetX < 0 || offsetY < 0, (double) row / (rows - 1), (double) col / (columns - 1));
              positionX += tileDimensions[0] - getPaddingX();
              inputIndex += 1;
            }
            positionY += rowHeight - getPaddingY();
            totalWidth = Math.max(totalWidth, positionX);
          }

          return outputData;
        }, inObj)).toArray(Tensor[]::new));
  }

  private int getInt(@Nonnull JsonObject json, String paddingX, int defaultValue) {
    JsonPrimitive asJsonPrimitive = json.getAsJsonPrimitive(paddingX);
    return null == asJsonPrimitive ? defaultValue : asJsonPrimitive.getAsInt();
  }

  @Nonnull
  private int[] getOutputDims(@Nonnull final Result[] inObj) {
    TensorList temp_63_0006 = inObj[0].getData();
    int[] dimensions1 = temp_63_0006.getDimensions();
    temp_63_0006.freeRef();
    int bands = dimensions1.length < 2 ? 1 : dimensions1[2];
    int totalWidth = 0;
    int totalHeight = 0;
    int inputIndex = 0;
    for (int row = 0; row < rows; row++) {
      int positionX = 0;
      int rowHeight = 0;
      for (int col = 0; col < columns; col++) {
        TensorList temp_63_0007 = inObj[inputIndex].getData();
        int[] dimensions = temp_63_0007.getDimensions();
        temp_63_0007.freeRef();
        rowHeight = Math.max(rowHeight, dimensions[1]);
        //positionX += dimensions[0] - positionX==0?0:getPaddingX();
        positionX += dimensions[0] - getPaddingX();
        inputIndex += 1;
      }
      //      totalHeight += rowHeight - totalHeight==0?0:getPaddingY();
      totalHeight += rowHeight - getPaddingY();
      totalWidth = Math.max(totalWidth, positionX);
    }
    RefUtil.freeRef(inObj);
    return new int[]{totalWidth + getPaddingX(), totalHeight + getPaddingY(), bands};
  }

  private static class Accumulator extends Result.Accumulator {

    private final Result[] inObj;
    private int paddingX;
    private int paddingY;
    private int columns;
    private int offsetX;
    private int offsetY;
    private int rows;

    /**
     * Instantiates a new Accumulator.
     *
     * @param columns  the columns
     * @param offsetX  the offset x
     * @param offsetY  the offset y
     * @param rows     the rows
     * @param paddingX the padding x
     * @param paddingY the padding y
     * @param inObj    the in obj
     */
    public Accumulator(int columns, int offsetX, int offsetY, int rows, int paddingX, int paddingY, Result... inObj) {
      this.inObj = inObj;
      this.paddingX = paddingX;
      this.paddingY = paddingY;
      this.columns = columns;
      this.offsetX = offsetX;
      this.offsetY = offsetY;
      this.rows = rows;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      final AtomicInteger positionY = new AtomicInteger(offsetX);
      int inputIndex = 0;
      for (int row = 0; row < rows; row++) {
        final AtomicInteger positionX = new AtomicInteger(offsetY);
        int rowHeight = 0;
        for (int col = 0; col < columns; col++) {
          Result in = inObj[inputIndex++].addRef();
          TensorList data = in.getData();
          int[] inputDataDimensions = data.getDimensions();
          data.freeRef();
          rowHeight = Math.max(rowHeight, inputDataDimensions[1]);
          if (in.isAlive()) {
            final int finalRow = row;
            final int finalCol = col;
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nullable final Tensor deltaTensor = delta.get(dataIndex);
                  @Nonnull final Tensor passbackTensor = new Tensor(inputDataDimensions);
                  ImgTileAssemblyLayer.copy(deltaTensor,
                      passbackTensor.addRef(), -positionX.get(),
                      -positionY.get(), 0 == positionX.get() ? 0 : paddingX / 2,
                      0 == positionY.get() ? 0 : paddingY / 2,
                      offsetX < 0 || offsetY < 0, (double) finalRow / rows, (double) finalCol / columns);
                  return passbackTensor;
                }, delta.addRef())).toArray(Tensor[]::new));
            DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
            Result.Accumulator accumulator = in.getAccumulator();
            try {
              accumulator.accept(buffer1, tensorArray);
            } finally {
              accumulator.freeRef();
            }
          }
          in.freeRef();
          positionX.addAndGet(inputDataDimensions[0] - paddingX);
        }
        positionY.addAndGet(rowHeight - paddingY);
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
    }
  }
}
