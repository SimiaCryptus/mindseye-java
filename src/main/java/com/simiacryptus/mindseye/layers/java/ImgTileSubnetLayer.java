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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Img tile subnet layer.
 */
@SuppressWarnings("serial")
public class ImgTileSubnetLayer extends WrapperLayer {

  private final int height;
  private final int width;
  private final int strideX;
  private final int strideY;

  /**
   * Instantiates a new Img tile subnet layer.
   *
   * @param subnetwork the subnetwork
   * @param width      the width
   * @param height     the height
   * @param strideX    the stride x
   * @param strideY    the stride y
   */
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height, final int strideX,
                            final int strideY) {
    super(subnetwork);
    this.height = height;
    this.width = width;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  /**
   * Instantiates a new Img tile subnet layer.
   *
   * @param subnetwork the subnetwork
   * @param width      the width
   * @param height     the height
   */
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height) {
    this(subnetwork, width, height, width, height);
  }

  /**
   * Instantiates a new Img tile subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgTileSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    height = json.getAsJsonPrimitive("height").getAsInt();
    width = json.getAsJsonPrimitive("width").getAsInt();
    strideX = json.getAsJsonPrimitive("strideX").getAsInt();
    strideY = json.getAsJsonPrimitive("strideY").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
  }

  /**
   * From json img tile subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img tile subnet layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgTileSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSubnetLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result input = inObj[0].addRef();
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    int cols = (int) (Math.ceil((inputDims[0] - width) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height) * 1.0 / strideY) + 1);
    if (cols == 1 && rows == 1) {
      input.freeRef();
      inputData.freeRef();
      return inner.eval(inObj);
    }
    RefUtil.freeRef(inObj);
    Result[] results = new Result[rows * cols];
    RefArrayList<TensorList> passback = new RefArrayList<TensorList>(cols * rows);
    for (int i = 0; i < cols * rows; i++) {
      passback.add(null);
    }
    int index = 0;
    AtomicInteger passbacks = new AtomicInteger(0);
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        int positionX = col * strideX;
        int positionY = row * strideY;
        assert positionX >= 0;
        assert positionY >= 0;
        assert positionX < inputDims[0];
        assert positionY < inputDims[1];
        final int finalIndex = index;
        Result.Accumulator accumulator = new TileAccumulator(passback.addRef(), finalIndex, passbacks, rows, cols, input.getAccumulator());
        ImgTileSelectLayer tileSelectLayer = new ImgTileSelectLayer(width, height, positionX, positionY);
        TensorList selectedTile = Result.getData(tileSelectLayer.eval(new Result(inputData.addRef())));
        tileSelectLayer.freeRef();
        RefUtil.set(results, index, inner.eval(new Result(selectedTile, accumulator)));
        index = index + 1;
      }
    }
    passback.freeRef();
    inputData.freeRef();
    input.freeRef();
    ImgTileAssemblyLayer imgTileAssemblyLayer = new ImgTileAssemblyLayer(cols, rows);
    Result assembledResult = imgTileAssemblyLayer.eval(results);
    imgTileAssemblyLayer.freeRef();
    return assembledResult;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("height", height);
    json.addProperty("width", width);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
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
  ImgTileSubnetLayer addRef() {
    return (ImgTileSubnetLayer) super.addRef();
  }

  private static class TileAccumulator extends Result.Accumulator {

    private final RefArrayList<TensorList> passback;
    private final int finalIndex;
    private final AtomicInteger passbacks;
    private final int rows;
    private final int cols;
    private Result.Accumulator accumulator;

    /**
     * Instantiates a new Tile accumulator.
     *
     * @param passback    the passback
     * @param finalIndex  the final index
     * @param passbacks   the passbacks
     * @param rows        the rows
     * @param cols        the cols
     * @param accumulator the accumulator
     */
    public TileAccumulator(RefArrayList<TensorList> passback, int finalIndex, AtomicInteger passbacks, int rows, int cols, Result.Accumulator accumulator) {
      this.passback = passback;
      this.finalIndex = finalIndex;
      this.passbacks = passbacks;
      this.rows = rows;
      this.cols = cols;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> ctx, @Nullable TensorList delta) {
      //Result.getData(tileSelectLayer.eval(new Result(delta)));
      RefUtil.freeRef(passback.set(finalIndex, delta));
      if (passbacks.incrementAndGet() == rows * cols) {
        passbacks.set(0);
        ImgTileAssemblyLayer imgTileAssemblyLayer = new ImgTileAssemblyLayer(cols, rows);
        TensorList reassembled = Result.getData(imgTileAssemblyLayer.eval(passback.stream().map(t -> {
          return new Result(t);
        }).<Result>toArray(Result[]::new)));
        imgTileAssemblyLayer.freeRef();
        this.accumulator.accept(ctx, reassembled);
      } else {
        if (null != ctx)
          ctx.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      RefUtil.freeRef(passback);
    }
  }
}
