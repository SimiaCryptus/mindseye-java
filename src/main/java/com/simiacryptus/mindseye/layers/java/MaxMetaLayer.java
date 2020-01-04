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
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class MaxMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxMetaLayer.class);

  public MaxMetaLayer() {
  }

  protected MaxMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static MaxMetaLayer fromJson(@Nonnull final JsonObject json,
                                      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new MaxMetaLayer(json);
  }

  public static @SuppressWarnings("unused")
  MaxMetaLayer[] addRefs(MaxMetaLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxMetaLayer::addRef)
        .toArray((x) -> new MaxMetaLayer[x]);
  }

  public static @SuppressWarnings("unused")
  MaxMetaLayer[][] addRefs(MaxMetaLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxMetaLayer::addRefs)
        .toArray((x) -> new MaxMetaLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0];
    final int itemCnt = input.getData().length();
    final Tensor input0Tensor = input.getData().get(0);
    final int vectorSize = input0Tensor.length();
    @Nonnull final int[] indicies = new int[vectorSize];
    for (int i = 0; i < vectorSize; i++) {
      final int itemNumber = i;
      indicies[i] = com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).mapToObj(x -> x)
          .max(com.simiacryptus.ref.wrappers.RefComparator.comparing(dataIndex -> {
            Tensor tensor = input.getData().get(dataIndex);
            return tensor.getData()[itemNumber];
          })).get();
    }
    return new Result(new TensorArray(input0Tensor.mapIndex((v, c) -> {
      Tensor tensor = input.getData().get(indicies[c]);
      return tensor.getData()[c];
    })), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (input.isAlive()) {
        @Nullable final Tensor delta = data.get(0);
        @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
        com.simiacryptus.ref.wrappers.RefArrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
        input0Tensor.coordStream(true).forEach((inputCoord) -> {
          feedback[indicies[inputCoord.getIndex()]].add(inputCoord, delta.get(inputCoord));
        });
        @Nonnull
        TensorArray tensorArray = new TensorArray(feedback);
        input.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive();
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

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MaxMetaLayer addRef() {
    return (MaxMetaLayer) super.addRef();
  }
}
