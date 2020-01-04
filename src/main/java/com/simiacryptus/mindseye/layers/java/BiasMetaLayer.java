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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class BiasMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasMetaLayer.class);

  public BiasMetaLayer() {
  }

  protected BiasMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static BiasMetaLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new BiasMetaLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0];
    final Result in1 = inObj[1];
    TensorList data0 = in0.getData();
    final int itemCnt = data0.length();
    final TensorList data1 = in1.getData();
    Tensor tensor1 = data1.get(0);
    final Tensor[] tensors = com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).parallel()
        .mapToObj(dataIndex -> {
          Tensor tensor = data0.get(dataIndex);
          return tensor.mapIndex((v, c) -> {
            return v + tensor1.get(c);
          });
        }).toArray(i -> new Tensor[i]);
    Tensor tensor0 = tensors[0];
    return new Result(new TensorArray(tensors),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          if (in1.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                com.simiacryptus.ref.wrappers.RefIntStream.range(0, data.length()).mapToObj(i -> {
                  if (i == 0)
                    return tensor0.mapCoords((c) -> {
                      return com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).mapToDouble(j -> {
                        Tensor tensor = data.get(j);
                        return tensor.get(c);
                      }).sum();
                    });
                  else {
                    return tensor0.mapCoords(v -> 0);
                  }
                }).toArray(i -> new Tensor[i]));
            in1.accumulate(buffer, tensorArray);
          }
          if (in0.isAlive()) {
            in0.accumulate(buffer, data);
          }
        }) {

      @Override
      public boolean isAlive() {
        return in0.isAlive() || in1.isAlive();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") BiasMetaLayer addRef() {
    return (BiasMetaLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") BiasMetaLayer[] addRefs(BiasMetaLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BiasMetaLayer::addRef)
        .toArray((x) -> new BiasMetaLayer[x]);
  }

  public static @SuppressWarnings("unused") BiasMetaLayer[][] addRefs(BiasMetaLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BiasMetaLayer::addRefs)
        .toArray((x) -> new BiasMetaLayer[x][]);
  }
}
