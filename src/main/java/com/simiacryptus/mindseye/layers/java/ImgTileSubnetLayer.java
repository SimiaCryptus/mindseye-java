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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

@SuppressWarnings("serial")
public @RefAware
class ImgTileSubnetLayer extends WrapperLayer {

  private final int height;
  private final int width;
  private final int strideX;
  private final int strideY;

  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height, final int strideX,
                            final int strideY) {
    super(subnetwork);
    if (null != subnetwork)
      subnetwork.freeRef();
    this.height = height;
    this.width = width;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height) {
    this(subnetwork, width, height, width, height);
    if (null != subnetwork)
      subnetwork.freeRef();
  }

  protected ImgTileSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    height = json.getAsJsonPrimitive("height").getAsInt();
    width = json.getAsJsonPrimitive("width").getAsInt();
    strideX = json.getAsJsonPrimitive("strideX").getAsInt();
    strideY = json.getAsJsonPrimitive("strideY").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
  }

  @SuppressWarnings("unused")
  public static ImgTileSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSubnetLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ImgTileSubnetLayer[] addRefs(ImgTileSubnetLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSubnetLayer::addRef)
        .toArray((x) -> new ImgTileSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgTileSubnetLayer[][] addRefs(ImgTileSubnetLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgTileSubnetLayer::addRefs)
        .toArray((x) -> new ImgTileSubnetLayer[x][]);
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
      if (null != input)
        input.freeRef();
      if (null != inputData)
        inputData.freeRef();
      Layer temp_12_0006 = getInner();
      Result temp_12_0005 = temp_12_0006
          .eval(Result.addRefs(inObj));
      if (null != temp_12_0006)
        temp_12_0006.freeRef();
      ReferenceCounting.freeRefs(inObj);
      return temp_12_0005;
    }
    ReferenceCounting.freeRefs(inObj);
    Result[] results = new Result[rows * cols];
    TensorList[] passback = new TensorList[rows * cols];
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
        ImgTileSelectLayer tileSelectLayer = new ImgTileSelectLayer(width, height, positionX, positionY);
        Result selectedTile = tileSelectLayer.eval(new Result(inputData, new Result.Accumulator() {
          {
          }

          @Override
          public void accept(DeltaSet<UUID> ctx, TensorList delta) {
            {
              TensorList temp_12_0001 = delta == null ? null : delta.addRef();
              if (null != passback[finalIndex])
                passback[finalIndex].freeRef();
              passback[finalIndex] = temp_12_0001 == null ? null : temp_12_0001.addRef();
              if (null != temp_12_0001)
                temp_12_0001.freeRef();
            }
            if (null != delta)
              delta.freeRef();
            if (passbacks.incrementAndGet() == rows * cols) {
              passbacks.set(0);
              ImgTileAssemblyLayer imgTileAssemblyLayer = new ImgTileAssemblyLayer(cols, rows);
              Result temp_12_0007 = imgTileAssemblyLayer
                  .eval(RefArrays.stream(TensorList.addRefs(passback)).map(t -> {
                    Result temp_12_0004 = new Result(t == null ? null : t.addRef(),
                        new Result.Accumulator() {
                          @Override
                          public void accept(DeltaSet<UUID> c2, TensorList d2) {
                            if (null != d2)
                              d2.freeRef();
                            if (null != c2)
                              c2.freeRef();
                          }

                          public @SuppressWarnings("unused")
                          void _free() {
                          }
                        });
                    if (null != t)
                      t.freeRef();
                    return temp_12_0004;
                  }).<Result>toArray(i -> new Result[i]));
              TensorList reassembled = temp_12_0007.getData();
              if (null != temp_12_0007)
                temp_12_0007.freeRef();
              if (null != imgTileAssemblyLayer)
                imgTileAssemblyLayer.freeRef();
              input.accumulate(ctx == null ? null : ctx.addRef(), reassembled == null ? null : reassembled.addRef());
              if (null != reassembled)
                reassembled.freeRef();
            }
            if (null != ctx)
              ctx.freeRef();
          }

          public @SuppressWarnings("unused")
          void _free() {
          }
        }) {
          public void _free() {
            super._free();
          }
        });
        if (null != tileSelectLayer)
          tileSelectLayer.freeRef();
        {
          Layer temp_12_0008 = getInner();
          Result temp_12_0002 = temp_12_0008
              .eval(selectedTile == null ? null : selectedTile.addRef());
          if (null != temp_12_0008)
            temp_12_0008.freeRef();
          if (null != results[index])
            results[index].freeRef();
          results[index] = temp_12_0002 == null ? null : temp_12_0002.addRef();
          if (null != temp_12_0002)
            temp_12_0002.freeRef();
        }
        if (null != selectedTile)
          selectedTile.freeRef();
        index = index + 1;
      }
    }
    if (null != passback)
      ReferenceCounting.freeRefs(passback);
    if (null != inputData)
      inputData.freeRef();
    if (null != input)
      input.freeRef();
    ImgTileAssemblyLayer imgTileAssemblyLayer = new ImgTileAssemblyLayer(cols, rows);
    Result temp_12_0003 = imgTileAssemblyLayer
        .eval(Result.addRefs(results));
    if (null != imgTileAssemblyLayer)
      imgTileAssemblyLayer.freeRef();
    if (null != results)
      ReferenceCounting.freeRefs(results);
    return temp_12_0003;
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
  }

  public @Override
  @SuppressWarnings("unused")
  ImgTileSubnetLayer addRef() {
    return (ImgTileSubnetLayer) super.addRef();
  }

}
