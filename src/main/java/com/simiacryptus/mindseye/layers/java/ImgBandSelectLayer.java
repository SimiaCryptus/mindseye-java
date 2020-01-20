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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class ImgBandSelectLayer extends LayerBase {

  private final int[] bands;

  public ImgBandSelectLayer(final int... bands) {
    super();
    this.bands = bands;
  }

  protected ImgBandSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    final JsonArray jsonArray = json.getAsJsonArray("bands");
    bands = new int[jsonArray.size()];
    for (int i = 0; i < bands.length; i++) {
      bands[i] = jsonArray.get(i).getAsInt();
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0].addRef();
    ReferenceCounting.freeRefs(inObj);
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    @Nonnull final Tensor outputDims = new Tensor(inputDims[0], inputDims[1], bands.length);
    @Nonnull
    TensorArray wrap = new TensorArray(RefIntStream.range(0, batch.length()).parallel().mapToObj(RefUtil
        .wrapInterface((IntFunction<? extends Tensor>) dataIndex -> outputDims.mapCoords(RefUtil.wrapInterface((c) -> {
              int[] coords = c.getCoords();
              @Nullable
              Tensor tensor = batch.get(dataIndex);
              double temp_45_0002 = tensor.get(coords[0], coords[1], bands[coords[2]]);
              tensor.freeRef();
              return temp_45_0002;
            }, batch.addRef())), outputDims,
            batch.addRef()))
        .toArray(i -> new Tensor[i]));
    batch.freeRef();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
          if (input.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nonnull final Tensor passback = new Tensor(inputDims);
                  @Nullable final Tensor err = error.get(dataIndex);
                  err.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
                    int[] coords = c.getCoords();
                    passback.set(coords[0], coords[1], bands[coords[2]], err.get(c));
                  }, passback.addRef(), err.addRef()));
                  err.freeRef();
                  return passback;
                }, error.addRef())).toArray(i -> new Tensor[i]));
            input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          error.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
        }
      };
      return new Result(wrap, accumulator) {
        {
          input.addRef();
        }
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }

        @Override
        public void _free() {
          input.freeRef();
          super._free();
        }
      };
    } finally {
      input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    @Nonnull final JsonArray array = new JsonArray();
    for (final int b : bands) {
      array.add(new JsonPrimitive(b));
    }
    json.add("bands", array);
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandSelectLayer addRef() {
    return (ImgBandSelectLayer) super.addRef();
  }

}
